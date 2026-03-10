"""CLI: Run live or historical replay multi-ticker simulation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import click

from config import DEVICE, ModelConfig, LiveConfig, TARGET_TICKERS
from src.data.preprocessor import load_scaler
from src.model.predictor import MultiPredictor
from src.live.runner import LiveRunner, ReplayRunner, LIVE_STRATEGIES
from src.live.dashboard import run_dashboard
from src.live.execution import DryRunExecutionClient, AlpacaExecutionClient, OrderRequest

_MC = ModelConfig()
_LC = LiveConfig()


def _run_replay_sweep(runner, n_configs: int, capital: float) -> None:
    """Run N replays with different thresholds, rank by Sharpe, save best."""
    thresholds = np.logspace(-2, -4, num=n_configs)  # 0.01 down to 0.0001
    results = []
    best_records = None
    best_sharpe = -np.inf

    for i, thresh in enumerate(thresholds):
        config = [{"name": f"Thresh_{thresh:.4f}", "long": float(thresh), "flat": float(-thresh)}]
        runner.reset(config)
        while not runner.is_done:
            runner.step()

        for name, ss in runner.state.strategies.items():
            ret = ss.current_return
            sharpe = ss.sharpe
            max_dd = ss.max_drawdown
            trades = ss.total_trades
            results.append({
                "threshold": thresh,
                "return": ret,
                "sharpe": sharpe,
                "max_dd": max_dd,
                "trades": trades,
            })
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_records = list(runner.state.records)
        print(f"  [SWEEP] {i + 1}/{n_configs} thresh={thresh:.4f} return={ret:+.2%} sharpe={sharpe:.2f}")

    results.sort(key=lambda r: r["sharpe"], reverse=True)
    print(f"\n{'='*70}")
    print("  REPLAY SWEEP RANKED BY SHARPE")
    print(f"{'='*70}")
    print(f"  {'#':<4} {'Threshold':>10} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7}")
    print(f"  {'-'*50}")
    for j, r in enumerate(results, 1):
        print(f"  {j:<4} {r['threshold']:>10.4f} {r['return']:>+9.2%} {r['sharpe']:>8.2f} {r['max_dd']:>+7.2%} {r['trades']:>7d}")
    print(f"{'='*70}")

    if best_records:
        import pandas as pd
        rows = [
            {"timestamp": r.timestamp, "ticker": r.ticker, "strategy": r.strategy,
             "action": r.action, "price": r.price, "signal": r.signal,
             "confidence": r.confidence, "agreement": r.agreement,
             "portfolio_value": r.portfolio_value, "position": r.position}
            for r in best_records
        ]
        pd.DataFrame(rows).to_csv("sweep_best_results.csv", index=False)
        print(f"\n[SWEEP] Best run saved to sweep_best_results.csv")


@click.command()
@click.option("--duration", default=_LC.duration_hours, help="Hours to run in live mode.", type=float)
@click.option("--interval", default=_LC.interval_minutes, help="Minutes between live data fetches.", type=int)
@click.option("--capital", default=_LC.initial_capital, help="Starting simulated capital per strategy.")
@click.option("--model-type", default=_MC.model_type, type=click.Choice(["lstm", "transformer"]))
@click.option("--seq-len", default=_MC.seq_len, help="Model lookback window.")
@click.option("--replay", is_flag=True, default=False, help="Run historical replay instead of live.")
@click.option("--replay-hours", default=24.0, type=float,
              help="Hours of history to replay (max: 168 for 1m, 1440 for 5m).")
@click.option("--replay-interval", default="1m", type=click.Choice(["1m", "2m", "5m", "15m", "30m", "1h"]),
              help="Candle interval for replay data.")
@click.option("--replay-speed", default=0.0, type=float,
              help="Seconds of wall time per replay tick (0 for instant batch mode).")
@click.option("--normalize-signal", is_flag=True, default=False,
              help="Use rolling z-score signal normalization (off = raw predictions, matches backtest).")
@click.option("--replay-sweep", default=0, type=int,
              help="When replaying: run N threshold configs, rank by Sharpe, save best (0=disabled).")
@click.option("--trade-output", is_flag=True, default=False,
              help="Show hourly limit buy/sell recommendations for manual trading.")
@click.option("--threshold", default=None, type=float,
              help="Use a single strategy with this long threshold (flat = -threshold).")
@click.option("--min-confidence", default=0.0, type=float,
              help="Trade-output/auto-trade gate: minimum confidence (0-1).")
@click.option("--min-signal", default=0.0005, type=float,
              help="Trade-output/auto-trade gate: minimum abs(signal) to act.")
@click.option("--auto-trade", is_flag=True, default=False,
              help="Place limit orders automatically from hourly recommendations.")
@click.option("--broker", default="dry-run", type=click.Choice(["dry-run", "alpaca-paper"]),
              help="Auto-trade broker adapter. alpaca-paper uses env vars (paper by default).")
@click.option("--max-orders-per-hour", default=4, type=int,
              help="Risk guard: max orders per hour (across all tickers).")
@click.option("--kill-loss", default=None, type=float,
              help="Risk guard: stop auto-trading if strategy return <= this (e.g. -0.05).")
def main(duration, interval, capital, model_type, seq_len,
         replay, replay_hours, replay_interval, replay_speed, normalize_signal, replay_sweep, trade_output,
         threshold, min_confidence, min_signal, auto_trade, broker, max_orders_per_hour, kill_loss):
    """Run live trading simulation or historical replay for UVXY, SPXU, SVIX, SPXL."""
    mcfg = ModelConfig(model_type=model_type, seq_len=seq_len)
    lcfg = LiveConfig(
        duration_hours=duration,
        interval_minutes=interval,
        initial_capital=capital,
        intraday_interval=f"{interval}m" if not replay else replay_interval,
    )

    feature_counts = {}
    for tkr in TARGET_TICKERS:
        try:
            scaler = load_scaler(ticker=tkr)
            feature_counts[tkr] = len(scaler.center_)
        except FileNotFoundError:
            print(f"ERROR: No scaler found for {tkr}. Train first: python scripts/train.py --ticker {tkr}")
            sys.exit(1)

    multi_predictor = MultiPredictor(
        feature_counts=feature_counts,
        tickers=TARGET_TICKERS,
        cfg=mcfg,
    )

    strategies = LIVE_STRATEGIES
    if threshold is not None:
        strategies = [{"name": f"Single_{threshold:.4f}", "long": float(threshold), "flat": float(-threshold)}]

    exec_client = None
    if auto_trade:
        if broker == "alpaca-paper":
            exec_client = AlpacaExecutionClient.from_env()
            print(f"[AUTO] Broker: alpaca ({exec_client.base_url})")
        else:
            exec_client = DryRunExecutionClient()
            print("[AUTO] Broker: dry-run")

    _last_hour = {"hour": None, "count": 0}

    def _on_hourly_recs(recs):
        if not auto_trade or exec_client is None:
            return
        if kill_loss is not None:
            try:
                # Use the first (or only) strategy as the live-trading performance proxy
                ss = next(iter(_runner_ref["runner"].state.strategies.values()))
                if ss.current_return <= float(kill_loss):
                    print(f"[AUTO] KILL-SWITCH: return {ss.current_return:+.2%} <= {kill_loss:+.2%}. No more orders.")
                    return
            except Exception:
                pass
        for r in recs:
            if _last_hour["hour"] != r.hour:
                _last_hour["hour"] = r.hour
                _last_hour["count"] = 0
            if _last_hour["count"] >= max_orders_per_hour:
                continue
            side = "buy" if "BUY" in r.action else "sell"
            req = OrderRequest(symbol=r.ticker, side=side, qty=r.quantity, limit_price=r.limit_price)
            res = exec_client.place_limit_order(req)
            if not res.ok:
                print(f"[AUTO] ERROR placing {r.ticker}: {res.error}")
            else:
                _last_hour["count"] += 1

    _runner_ref = {"runner": None}

    if replay:
        runner = ReplayRunner(
            multi_predictor=multi_predictor,
            live_cfg=lcfg,
            model_cfg=mcfg,
            strategies=strategies,
            replay_hours=replay_hours,
            replay_interval=replay_interval,
            normalize_signal=normalize_signal,
            trade_output=trade_output,
            trade_min_signal=min_signal,
            trade_min_confidence=min_confidence,
            on_hourly_recs=_on_hourly_recs,
        )
        _runner_ref["runner"] = runner
        if replay_sweep > 0:
            _run_replay_sweep(runner, replay_sweep, lcfg.initial_capital)
        else:
            run_dashboard(runner, sleep_seconds=replay_speed)
    else:
        runner = LiveRunner(
            multi_predictor=multi_predictor,
            live_cfg=lcfg,
            model_cfg=mcfg,
            strategies=strategies,
            normalize_signal=normalize_signal,
            trade_output=trade_output,
            trade_min_signal=min_signal,
            trade_min_confidence=min_confidence,
            on_hourly_recs=_on_hourly_recs,
        )
        _runner_ref["runner"] = runner
        run_dashboard(runner)


if __name__ == "__main__":
    main()
