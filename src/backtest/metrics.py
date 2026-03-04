"""Performance metrics and statistical significance testing for backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    label: str
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    num_trades: int
    benchmark_return: float


def compute_metrics(
    equity_curve: pd.Series,
    trade_returns: List[float],
    benchmark_curve: pd.Series,
    periods_per_year: float = 252.0,
    label: str = "Strategy",
) -> BacktestMetrics:
    """Compute comprehensive performance metrics."""

    daily_rets = equity_curve.pct_change().dropna()
    n_days = len(daily_rets)

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    ann_factor = periods_per_year / max(n_days, 1)
    annualized_return = (1 + total_return) ** ann_factor - 1
    annualized_vol = daily_rets.std() * np.sqrt(periods_per_year) if n_days > 1 else 0.0

    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0.0

    downside = daily_rets[daily_rets < 0]
    downside_std = downside.std() * np.sqrt(periods_per_year) if len(downside) > 1 else 0.0
    sortino = annualized_return / downside_std if downside_std > 0 else 0.0

    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()

    in_dd = drawdown < 0
    dd_groups = (~in_dd).cumsum()
    dd_durations = in_dd.groupby(dd_groups).sum()
    max_dd_dur = int(dd_durations.max()) if len(dd_durations) > 0 else 0

    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]
    win_rate = len(wins) / max(len(trade_returns), 1)
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0

    bench_return = (benchmark_curve.iloc[-1] / benchmark_curve.iloc[0]) - 1

    return BacktestMetrics(
        label=label,
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration_days=max_dd_dur,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        num_trades=len(trade_returns),
        benchmark_return=bench_return,
    )


def print_metrics(m: BacktestMetrics) -> None:
    """Print a formatted summary table."""
    print(f"\n{'=' * 58}")
    print(f"  {m.label}")
    print(f"{'=' * 58}")
    print(f"  Total Return:          {m.total_return:>+10.2%}")
    print(f"  Annualized Return:     {m.annualized_return:>+10.2%}")
    print(f"  Annualized Volatility: {m.annualized_volatility:>10.2%}")
    print(f"  Sharpe Ratio:          {m.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:         {m.sortino_ratio:>10.2f}")
    print(f"  Max Drawdown:          {m.max_drawdown:>+10.2%}")
    print(f"  Max DD Duration:       {m.max_drawdown_duration_days:>7d} days")
    print(f"{'-' * 58}")
    print(f"  Trades:                {m.num_trades:>10d}")
    print(f"  Win Rate:              {m.win_rate:>10.1%}")
    print(f"  Profit Factor:         {m.profit_factor:>10.2f}")
    print(f"  Avg Win:               {m.avg_win:>+10.4f}")
    print(f"  Avg Loss:              {m.avg_loss:>+10.4f}")
    print(f"{'-' * 58}")
    print(f"  Benchmark (B&H):       {m.benchmark_return:>+10.2%}")
    excess = m.total_return - m.benchmark_return
    print(f"  Excess Return:         {excess:>+10.2%}")
    print(f"{'=' * 58}")


def print_comparison_table(results: List[BacktestMetrics]) -> None:
    """Print a side-by-side comparison of multiple strategy results."""
    print(f"\n{'='*100}")
    print(f"  STRATEGY COMPARISON")
    print(f"{'='*100}")

    header = f"  {'Strategy':<22} {'Return':>9} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>9} {'Trades':>7} {'WinRate':>8} {'PF':>6} {'Excess':>9}"
    print(header)
    print(f"  {'-'*94}")

    for m in results:
        excess = m.total_return - m.benchmark_return
        pf = f"{m.profit_factor:.2f}" if m.profit_factor < 100 else "inf"
        print(
            f"  {m.label:<22} {m.total_return:>+8.1%} {m.sharpe_ratio:>8.2f} "
            f"{m.sortino_ratio:>8.2f} {m.max_drawdown:>+8.1%} {m.num_trades:>7d} "
            f"{m.win_rate:>7.1%} {pf:>6} {excess:>+8.1%}"
        )

    print(f"  {'-'*94}")
    bench = results[0].benchmark_return if results else 0
    print(f"  {'Buy & Hold ETHU':<22} {bench:>+8.1%} {'---':>8} {'---':>8} {'---':>9} {'---':>7} {'---':>8} {'---':>6} {'---':>9}")
    print(f"{'='*100}")


def monte_carlo_significance(
    actual_return: float,
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    prices,
    engine,
    n_simulations: int = 500,
) -> dict:
    """Test if strategy return is statistically significant vs random.

    Shuffles the prediction signs randomly and re-runs the backtest
    many times to build a null distribution.
    """
    from src.backtest.engine import BacktestEngine

    random_returns = []
    rng = np.random.default_rng(42)

    for _ in range(n_simulations):
        shuffled = predictions.copy()
        rng.shuffle(shuffled)
        eq, trades, _ = engine.run(shuffled, actual_returns, prices)
        r = (eq.iloc[-1] / eq.iloc[0]) - 1
        random_returns.append(r)

    random_returns = np.array(random_returns)
    p_value = np.mean(random_returns >= actual_return)
    percentile = np.mean(random_returns < actual_return) * 100

    return {
        "p_value": p_value,
        "percentile": percentile,
        "mean_random": np.mean(random_returns),
        "std_random": np.std(random_returns),
        "median_random": np.median(random_returns),
        "n_simulations": n_simulations,
    }
