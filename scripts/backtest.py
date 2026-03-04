"""CLI: Comprehensive backtesting with multi-strategy grid and Monte Carlo significance."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import numpy as np
import torch

from config import DEVICE, MODELS_DIR, ModelConfig, BacktestConfig, TARGET_TICKER
from src.data.storage import read_all, init_db
from src.data.preprocessor import build_features, load_scaler, load_feature_cols
from src.model.architecture import build_model
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import (
    compute_metrics, print_metrics, print_comparison_table,
    monte_carlo_significance,
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
    models = []
    for p in paths:
        m = build_model(n_features, mcfg)
        ckpt = torch.load(p, map_location=DEVICE, weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models.append(m)
    print(f"[BACKTEST] Loaded {len(models)} model(s)")
    return models


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


@click.command()
@click.option("--model-path", default=None, help="Path to model checkpoint.")
@click.option("--model-type", default=_MC.model_type, type=click.Choice(["lstm", "transformer"]))
@click.option("--seq-len", default=_MC.seq_len)
@click.option("--start", default=None, help="Backtest start date (YYYY-MM-DD).")
@click.option("--capital", default=_BC.initial_capital)
@click.option("--monte-carlo", default=500, help="Number of Monte Carlo simulations (0 to skip).")
@click.option("--grid/--no-grid", default=True, help="Run multi-strategy grid.")
def main(model_path, model_type, seq_len, start, capital, monte_carlo, grid):
    """Run comprehensive backtest with strategy grid and significance testing."""
    init_db()
    print("[BACKTEST] Loading data...")
    merged = read_all()
    if merged.empty:
        print("ERROR: No data. Run `python scripts/fetch_data.py` first.")
        sys.exit(1)

    feat_df, target_col = build_features(merged)

    if start:
        feat_df = feat_df[feat_df.index >= start]

    try:
        saved_cols = load_feature_cols()
        available = set(c for c in feat_df.columns if c != target_col)
        for col in saved_cols:
            if col not in available:
                feat_df[col] = 0.0
        feature_cols = saved_cols
    except FileNotFoundError:
        feature_cols = [c for c in feat_df.columns if c != target_col]

    X_all = feat_df[feature_cols].values.astype(np.float32)
    y_all = feat_df[target_col].values.astype(np.float32)

    scaler = load_scaler()
    X_scaled = scaler.transform(X_all)

    mcfg = ModelConfig(model_type=model_type, seq_len=seq_len)
    mp = Path(model_path) if model_path else MODELS_DIR / "best_model.pt"
    n_features = X_scaled.shape[1]

    models = _load_ensemble(mp, n_features, mcfg)

    print("[BACKTEST] Generating predictions...")
    predictions = _generate_predictions(models, X_scaled, seq_len)

    actual = y_all[seq_len - 1 : seq_len - 1 + len(predictions)]
    ethu_close = merged[(TARGET_TICKER, "close")].reindex(feat_df.index)
    prices = ethu_close.iloc[seq_len - 1 : seq_len - 1 + len(predictions)].copy()
    prices = prices.iloc[: len(predictions)]

    min_len = min(len(predictions), len(actual), len(prices))
    predictions = predictions[:min_len]
    actual = actual[:min_len]
    prices = prices.iloc[:min_len]

    dir_acc = np.mean((predictions > 0) == (actual > 0))
    print(f"\n  Directional Accuracy: {dir_acc:.1%}")
    print(f"  Prediction periods:   {len(predictions)}")
    print(f"  Ensemble size:        {len(models)}")

    # --- Multi-strategy grid ---
    if grid:
        strategies = STRATEGY_GRID
    else:
        strategies = [{"label": "Default", "long_threshold": 0.001, "flat_threshold": -0.001, "slippage_bps": 5.0}]

    all_metrics = []
    default_equity = None
    default_engine = None

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

        if not grid:
            print_metrics(m)

    if grid:
        print_comparison_table(all_metrics)

        best = max(all_metrics, key=lambda x: x.sharpe_ratio)
        print(f"\n  Best Sharpe: {best.label} ({best.sharpe_ratio:.2f})")
        print_metrics(best)

    # --- Monte Carlo significance test ---
    if monte_carlo > 0 and default_engine is not None:
        print(f"\n{'='*58}")
        print(f"  MONTE CARLO SIGNIFICANCE TEST  ({monte_carlo} simulations)")
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
            print("  The strategy's return could be explained by random chance.")

    print(f"\n{'='*58}")


if __name__ == "__main__":
    main()
