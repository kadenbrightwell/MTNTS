"""CLI: Comprehensive backtesting with per-ticker strategy grids, Monte Carlo, and portfolio summary."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import concurrent.futures

import click
import numpy as np
import torch

from config import (
    DEVICE, ModelConfig, BacktestConfig,
    TARGET_TICKERS, ticker_model_dir,
)
from src.data.storage import read_all, init_db
from src.data.preprocessor import build_features, load_scaler, load_feature_cols
from src.model.architecture import build_model, detect_checkpoint_config
from src.backtest.engine import BacktestEngine, MultiTickerEngine
from src.backtest.metrics import (
    compute_metrics, print_metrics, print_comparison_table,
    print_multi_ticker_summary, monte_carlo_significance,
)

_MC = ModelConfig()
_BC = BacktestConfig()

STRATEGY_GRID = [
    {"label": "Conservative",   "long_threshold": 0.005,  "flat_threshold": -0.005,  "slippage_bps": 10.0},
    {"label": "Moderate",       "long_threshold": 0.002,  "flat_threshold": -0.002,  "slippage_bps": 5.0},
    {"label": "Default",        "long_threshold": 0.001,  "flat_threshold": -0.001,  "slippage_bps": 5.0},
    {"label": "Aggressive",     "long_threshold": 0.0005, "flat_threshold": -0.0005, "slippage_bps": 5.0},
    {"label": "Ultra-Aggr",     "long_threshold": 0.0,    "flat_threshold": 0.0,     "slippage_bps": 5.0},
    {"label": "Low Cost",       "long_threshold": 0.001,  "flat_threshold": -0.001,  "slippage_bps": 1.0},
    {"label": "High Cost",      "long_threshold": 0.001,  "flat_threshold": -0.001,  "slippage_bps": 15.0},
]


def _load_ensemble(base_path: Path, n_features: int, mcfg: ModelConfig):
    stem, suffix = base_path.stem, base_path.suffix
    parent = base_path.parent
    ensemble_paths = sorted(parent.glob(f"{stem}_seed*{suffix}"))
    paths = ensemble_paths if ensemble_paths else ([base_path] if base_path.exists() else [])
    if not paths:
        print(f"ERROR: No model found at {base_path}")
        sys.exit(1)

    first_ckpt = torch.load(paths[0], map_location=DEVICE, weights_only=False)
    ckpt_info = detect_checkpoint_config(first_ckpt)
    detected_type = ckpt_info["model_type"]
    detected_seq = ckpt_info.get("seq_len", mcfg.seq_len)
    if detected_type != mcfg.model_type or detected_seq != mcfg.seq_len:
        print(f"[BACKTEST] Auto-detected: model={detected_type}, seq_len={detected_seq}")
        mcfg = ModelConfig(model_type=detected_type, seq_len=detected_seq,
                           hidden_size=mcfg.hidden_size, num_layers=mcfg.num_layers,
                           num_heads=mcfg.num_heads, dropout=mcfg.dropout,
                           fc_hidden=mcfg.fc_hidden, dim_feedforward=mcfg.dim_feedforward)

    models = []
    for i, p in enumerate(paths):
        ckpt = first_ckpt if i == 0 else torch.load(p, map_location=DEVICE, weights_only=False)
        m = build_model(n_features, mcfg)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models.append(m)
    print(f"[BACKTEST] Loaded {len(models)} {detected_type} model(s) (seq_len={mcfg.seq_len})")
    return models, mcfg


def _generate_predictions(models, X_scaled, seq_len):
    predictions = []
    for i in range(len(X_scaled) - seq_len):
        window = X_scaled[i : i + seq_len]
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        preds = []
        with torch.no_grad():
            for m in models:
                with torch.amp.autocast(DEVICE.type, enabled=(DEVICE.type == "cuda")):
                    preds.append(m(x).item())
        predictions.append(float(np.mean(preds)))
    return np.array(predictions)


def _backtest_single_ticker(target_ticker, merged, mcfg, capital, monte_carlo, grid, start, model_path):
    """Run a full backtest for one target ticker. Returns the Default-strategy metrics."""
    print(f"\n{'#'*65}")
    print(f"  BACKTEST: {target_ticker}")
    print(f"{'#'*65}")

    feat_df, target_col = build_features(merged, target_ticker=target_ticker)

    if start:
        feat_df = feat_df[feat_df.index >= start]

    try:
        saved_cols = load_feature_cols(ticker=target_ticker)
        available = set(c for c in feat_df.columns if c != target_col)
        for col in saved_cols:
            if col not in available:
                feat_df[col] = 0.0
        feature_cols = saved_cols
    except FileNotFoundError:
        feature_cols = [c for c in feat_df.columns if c != target_col]

    X_all = feat_df[feature_cols].values.astype(np.float32)
    y_all = feat_df[target_col].values.astype(np.float32)

    scaler = load_scaler(ticker=target_ticker)
    X_scaled = scaler.transform(X_all)

    if model_path:
        mp = Path(model_path)
    else:
        mp = ticker_model_dir(target_ticker) / "best_model.pt"

    n_features = X_scaled.shape[1]
    models, mcfg = _load_ensemble(mp, n_features, mcfg)
    seq_len = mcfg.seq_len

    print(f"[BACKTEST] [{target_ticker}] Generating predictions...")
    predictions = _generate_predictions(models, X_scaled, seq_len)

    actual = y_all[seq_len - 1 : seq_len - 1 + len(predictions)]
    target_close = merged[(target_ticker, "close")].reindex(feat_df.index)
    prices = target_close.iloc[seq_len - 1 : seq_len - 1 + len(predictions)].copy()
    prices = prices.iloc[: len(predictions)]

    min_len = min(len(predictions), len(actual), len(prices))
    predictions = predictions[:min_len]
    actual = actual[:min_len]
    prices = prices.iloc[:min_len]

    dir_acc = np.mean((predictions > 0) == (actual > 0))
    print(f"\n  [{target_ticker}] Directional Accuracy: {dir_acc:.1%}")
    print(f"  [{target_ticker}] Prediction periods:   {len(predictions)}")
    print(f"  [{target_ticker}] Ensemble size:        {len(models)}")

    strategies = STRATEGY_GRID if grid else [
        {"label": "Default", "long_threshold": 0.001, "flat_threshold": -0.001, "slippage_bps": 5.0}
    ]

    all_metrics = []
    default_equity = None
    default_engine = None
    default_metrics = None

    for strat in strategies:
        bcfg = BacktestConfig(
            initial_capital=capital,
            slippage_bps=strat["slippage_bps"],
            commission_bps=strat.get("commission_bps", 5.0),
            long_threshold=strat["long_threshold"],
            flat_threshold=strat["flat_threshold"],
        )
        engine = BacktestEngine(bcfg)
        eq, trades, bench = engine.run(predictions, actual, prices)
        m = compute_metrics(eq, trades, bench, label=strat["label"])
        all_metrics.append(m)

        if strat["label"] == "Default":
            default_equity = eq
            default_engine = engine
            default_metrics = m

        if not grid:
            print_metrics(m)

    if grid:
        print_comparison_table(all_metrics, ticker=target_ticker)

        best = max(all_metrics, key=lambda x: x.sharpe_ratio)
        print(f"\n  [{target_ticker}] Best Sharpe: {best.label} ({best.sharpe_ratio:.2f})")
        print_metrics(best)

    if monte_carlo > 0 and default_engine is not None:
        print(f"\n{'='*58}")
        print(f"  MONTE CARLO  [{target_ticker}]  ({monte_carlo} simulations)")
        print(f"{'='*58}")

        default_return = (default_equity.iloc[-1] / default_equity.iloc[0]) - 1 if default_equity is not None else 0

        mc = monte_carlo_significance(
            actual_return=default_return,
            predictions=predictions,
            actual_returns=actual,
            prices=prices,
            engine=default_engine,
            n_simulations=monte_carlo,
        )

        print(f"  Strategy return:     {default_return:>+10.2%}")
        print(f"  Random mean return:  {mc['mean_random']:>+10.2%}")
        print(f"  Random median:       {mc['median_random']:>+10.2%}")
        print(f"  Random std:          {mc['std_random']:>10.2%}")
        print(f"  Percentile rank:     {mc['percentile']:>9.1f}th")
        print(f"  p-value:             {mc['p_value']:>10.4f}")

        if mc["p_value"] < 0.05:
            print(f"\n  Result: SIGNIFICANT at 5% level (p={mc['p_value']:.4f})")
        elif mc["p_value"] < 0.10:
            print(f"\n  Result: MARGINAL significance (p={mc['p_value']:.4f})")
        else:
            print(f"\n  Result: NOT significant (p={mc['p_value']:.4f})")

    return default_metrics


def _backtest_worker(args):
    """Worker for parallel backtesting. Returns (ticker, metrics, equity_curve) or (ticker, error)."""
    tkr, merged, mcfg, capital, monte_carlo, grid, start, model_path = args
    try:
        m = _backtest_single_ticker(
            tkr, merged, mcfg, capital, monte_carlo, grid, start, model_path
        )
        return tkr, m, None
    except Exception as e:
        return tkr, None, str(e)


@click.command()
@click.option("--model-path", default=None, help="Path to model checkpoint (overrides per-ticker default).")
@click.option("--model-type", default=_MC.model_type, type=click.Choice(["lstm", "transformer"]))
@click.option("--seq-len", default=_MC.seq_len)
@click.option("--start", default=None, help="Backtest start date (YYYY-MM-DD).")
@click.option("--capital", default=_BC.initial_capital)
@click.option("--monte-carlo", default=500, help="Number of Monte Carlo simulations (0 to skip).")
@click.option("--grid/--no-grid", default=True, help="Run multi-strategy grid.")
@click.option("--ticker", default=None, help="Backtest a single ticker (e.g. UVXY). Default: all 4.")
@click.option("--workers", default=1, help="Parallel workers (1=serial, >1=parallel tickers).")
def main(model_path, model_type, seq_len, start, capital, monte_carlo, grid, ticker, workers):
    """Run backtests for UVXY, SPXU, SVIX, SPXL with strategy grids and significance testing."""
    init_db()
    print("[BACKTEST] Loading data...")
    merged = read_all()
    if merged.empty:
        print("ERROR: No data. Run `python scripts/fetch_data.py` first.")
        sys.exit(1)

    mcfg = ModelConfig(model_type=model_type, seq_len=seq_len)
    tickers_to_test = [ticker] if ticker else TARGET_TICKERS

    per_ticker_capital = capital / len(tickers_to_test)
    per_ticker_metrics = {}

    if workers > 1 and len(tickers_to_test) > 1:
        print(f"[BACKTEST] Running {len(tickers_to_test)} tickers with {workers} workers...")
        args_list = [
            (tkr, merged, mcfg, per_ticker_capital, monte_carlo, grid, start, model_path)
            for tkr in tickers_to_test
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_backtest_worker, a): a[0] for a in args_list}
            for future in concurrent.futures.as_completed(futures):
                tkr, m, err = future.result()
                if err:
                    print(f"\n  WARNING: Backtest failed for {tkr}: {err}")
                elif m is not None:
                    per_ticker_metrics[tkr] = m
    else:
        for tkr in tickers_to_test:
            try:
                m = _backtest_single_ticker(
                    tkr, merged, mcfg, per_ticker_capital, monte_carlo, grid, start, model_path
                )
                if m is not None:
                    per_ticker_metrics[tkr] = m
            except Exception as e:
                print(f"\n  WARNING: Backtest failed for {tkr}: {e}")

    if len(per_ticker_metrics) > 1:
        print_multi_ticker_summary(per_ticker_metrics, total_capital=capital)

    print(f"\n{'='*58}")


if __name__ == "__main__":
    main()
