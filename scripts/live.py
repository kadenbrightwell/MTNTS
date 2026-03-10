"""CLI: Run live or historical replay multi-ticker simulation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click

from config import DEVICE, ModelConfig, LiveConfig, TARGET_TICKERS
from src.data.preprocessor import load_scaler
from src.model.predictor import MultiPredictor
from src.live.runner import LiveRunner, ReplayRunner, LIVE_STRATEGIES
from src.live.dashboard import run_dashboard

_MC = ModelConfig()
_LC = LiveConfig()


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
def main(duration, interval, capital, model_type, seq_len,
         replay, replay_hours, replay_interval, replay_speed):
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

    if replay:
        runner = ReplayRunner(
            multi_predictor=multi_predictor,
            live_cfg=lcfg,
            model_cfg=mcfg,
            strategies=LIVE_STRATEGIES,
            replay_hours=replay_hours,
            replay_interval=replay_interval,
        )
        run_dashboard(runner, sleep_seconds=replay_speed)
    else:
        runner = LiveRunner(
            multi_predictor=multi_predictor,
            live_cfg=lcfg,
            model_cfg=mcfg,
            strategies=LIVE_STRATEGIES,
        )
        run_dashboard(runner)


if __name__ == "__main__":
    main()
