"""Rich + plotext terminal dashboard for live and replay trading simulations."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Union

import plotext as plt
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config import DEVICE

if TYPE_CHECKING:
    from src.live.runner import LiveRunner, ReplayRunner


STRAT_COLORS = {
    "Conservative": "bright_red",
    "Moderate":     "bright_yellow",
    "Default":      "bright_green",
    "Aggressive":   "bright_cyan",
    "Ultra-Aggr":   "bright_magenta",
}

PLOT_COLORS = {
    "Conservative": "red",
    "Moderate":     "yellow",
    "Default":      "green",
    "Aggressive":   "cyan",
    "Ultra-Aggr":   "magenta",
}


def _build_chart(runner, width: int = 110, height: int = 20) -> str:
    st = runner.state
    if len(st.prices) < 2:
        return "  Waiting for data points..."

    plt.clf()
    plt.plotsize(width, height)
    plt.theme("dark")

    p0 = st.prices[0] if st.prices[0] != 0 else 1
    x = list(range(len(st.prices)))
    norm_prices = [p / p0 * 100 for p in st.prices]
    plt.plot(x, norm_prices, label="ETHU", color="white")

    for name, ss in st.strategies.items():
        if len(ss.portfolio_values) < 2:
            continue
        v0 = ss.portfolio_values[0] if ss.portfolio_values[0] != 0 else 1
        vals = [v / v0 * 100 for v in ss.portfolio_values[:len(x) + 1]]
        if len(vals) > len(x):
            vals = vals[:len(x)]
        elif len(vals) < len(x):
            continue
        color = PLOT_COLORS.get(name, "blue")
        plt.plot(x[:len(vals)], vals, label=name, color=color)

    plt.title("Normalised Performance (base=100)")
    plt.xlabel("Ticks")
    plt.ylabel("%")

    return plt.build()


def _build_confidence_bar(confidence: float, agreement: float) -> Text:
    bar_len = 20
    filled = int(confidence * bar_len)
    bar = Text()

    if confidence > 0.7:
        style = "bold green"
    elif confidence > 0.4:
        style = "bold yellow"
    else:
        style = "bold red"

    bar.append("[")
    bar.append("=" * filled, style=style)
    bar.append("-" * (bar_len - filled), style="dim")
    bar.append(f"] {confidence:.0%}", style=style)
    bar.append(f"  ({agreement:.0%} agree)", style="dim white")
    return bar


def _build_strategy_table(runner) -> Table:
    st = runner.state
    tbl = Table(title="Strategy Portfolios", show_header=True, border_style="bright_blue", padding=(0, 1))
    tbl.add_column("Strategy", style="bold", width=14)
    tbl.add_column("Value", justify="right", width=12)
    tbl.add_column("Return", justify="right", width=9)
    tbl.add_column("Trades", justify="center", width=7)
    tbl.add_column("Win%", justify="right", width=7)
    tbl.add_column("MaxDD", justify="right", width=8)
    tbl.add_column("Sharpe", justify="right", width=7)
    tbl.add_column("Pos", justify="center", width=6)

    for name, ss in st.strategies.items():
        val = ss.portfolio_values[-1] if ss.portfolio_values else 0
        ret = ss.current_return
        ret_style = STRAT_COLORS.get(name, "white")
        pos = "LONG" if ss.position == 1 else "FLAT"
        pos_style = "bold green" if ss.position == 1 else "dim"
        wr = f"{ss.win_rate:.0%}" if ss.trade_returns else "---"
        dd = f"{ss.max_drawdown:+.1%}" if ss.max_drawdown < 0 else "0.0%"

        tbl.add_row(
            Text(name, style=ret_style),
            f"${val:,.2f}",
            Text(f"{ret:+.2%}", style="green" if ret >= 0 else "red"),
            str(ss.num_trades),
            wr,
            Text(dd, style="red" if ss.max_drawdown < -0.02 else "dim"),
            f"{ss.sharpe:.2f}",
            Text(pos, style=pos_style),
        )

    if st.prices:
        bench_ret = st.price_return
        tbl.add_row(
            Text("Buy & Hold", style="dim white"),
            "---",
            Text(f"{bench_ret:+.2%}", style="cyan" if bench_ret >= 0 else "red"),
            "---", "---", "---", "---",
            Text("LONG", style="dim"),
        )

    return tbl


def _build_signal_panel(runner) -> Panel:
    st = runner.state
    if not st.signals:
        return Panel("Waiting for first prediction...", title="Signal", border_style="blue")

    lines = Text()
    sig = st.signals[-1]
    conf = st.confidences[-1] if st.confidences else 0
    agree = st.agreements[-1] if st.agreements else 0

    if sig > 0.001:
        sig_style = "bold green"
        direction = "BULLISH"
    elif sig < -0.001:
        sig_style = "bold red"
        direction = "BEARISH"
    else:
        sig_style = "dim"
        direction = "NEUTRAL"

    lines.append(f"  Signal: {sig:+.6f}  ", style=sig_style)
    lines.append(f"[{direction}]\n", style=sig_style)
    lines.append("  Confidence: ")
    lines.append_text(_build_confidence_bar(conf, agree))
    lines.append("\n")

    if st.per_model_preds:
        last_preds = st.per_model_preds[-1]
        n_bull = sum(1 for p in last_preds if p > 0)
        n_bear = len(last_preds) - n_bull
        lines.append(f"  Models: {n_bull} bull / {n_bear} bear  ", style="white")
        lines.append(f"std={st.pred_stds[-1]:.6f}", style="dim")

    return Panel(lines, title="[bright_white]Ensemble Signal[/bright_white]", border_style="bright_blue")


def _build_trade_log(runner, max_rows: int = 6) -> Panel:
    recs = [r for r in runner.state.records if r.action != "HOLD"]
    recent = recs[-max_rows:] if recs else []

    if not recent:
        return Panel("  No trades yet.", title="Recent Trades", border_style="blue")

    tbl = Table(show_header=True, border_style="dim", padding=(0, 1))
    tbl.add_column("Time", width=9)
    tbl.add_column("Strategy", width=14)
    tbl.add_column("Action", width=6)
    tbl.add_column("Price", justify="right", width=10)
    tbl.add_column("Conf", justify="right", width=6)

    for r in recent:
        act_style = "bold green" if r.action == "BUY" else "bold red"
        ts_display = r.timestamp.split(" ")[-1][:8] if " " in r.timestamp else r.timestamp[:8]
        tbl.add_row(
            ts_display,
            Text(r.strategy, style=STRAT_COLORS.get(r.strategy, "white")),
            Text(r.action, style=act_style),
            f"${r.price:.2f}",
            f"{r.confidence:.0%}",
        )

    return Panel(tbl, title="[bright_white]Recent Trades[/bright_white]", border_style="bright_blue")


def _build_header(runner) -> Text:
    import torch
    elapsed_str = str(runner.elapsed).split(".")[0]

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    model_name = runner.mcfg.model_type.upper()
    n_models = len(runner.predictor.models)
    st = runner.state
    fetches = f"{st.fetch_successes}/{st.fetch_successes + st.fetch_errors}"

    header = Text()

    if runner.mode == "REPLAY":
        progress = runner.tick_progress
        header.append(f" REPLAY", style="bold bright_magenta")
        header.append(f"  tick: {progress}", style="bold white")
        header.append(f"  wall: {elapsed_str}", style="dim white")
    else:
        total = dt.timedelta(hours=runner.lcfg.duration_hours)
        total_str = str(total).split(".")[0]
        remaining = str(runner.remaining).split(".")[0]
        header.append(f" {elapsed_str} / {total_str}", style="bold white")
        header.append(f"  rem: {remaining}", style="dim white")

    header.append(f"    {model_name} x{n_models}", style="bold cyan")
    header.append(f"    {gpu}", style="bold yellow")
    header.append(f"    fetches: {fetches}", style="dim")
    return header


def build_display(runner) -> Panel:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="chart", size=22),
        Layout(name="middle", size=12),
        Layout(name="bottom", size=10),
    )

    layout["header"].update(
        Panel(_build_header(runner), style="bold", border_style="bright_blue")
    )

    chart_str = _build_chart(runner)
    layout["chart"].update(
        Panel(chart_str, title="[cyan]Multi-Strategy Performance[/cyan]", border_style="blue")
    )

    layout["middle"].update(_build_strategy_table(runner))

    layout["bottom"].split_row(
        Layout(name="signal", ratio=1),
        Layout(name="trades", ratio=1),
    )
    layout["bottom"]["signal"].update(_build_signal_panel(runner))
    layout["bottom"]["trades"].update(_build_trade_log(runner))

    mode_label = "REPLAY" if runner.mode == "REPLAY" else "LIVE"
    return Panel(
        layout,
        title=f"[bold bright_white] ETHU Neural Trader  -  Multi-Strategy {mode_label} [/bold bright_white]",
        border_style="bright_blue",
        padding=(0, 1),
    )


def run_dashboard(runner, sleep_seconds: float | None = None) -> None:
    """Main loop for both live and replay modes.

    For live mode, sleep_seconds defaults to interval_minutes * 60.
    For replay mode, sleep_seconds defaults to 1.0 (1 second per tick).
    """
    console = Console()
    _ = runner.start_time

    n_models = len(runner.predictor.models)
    n_strats = len(runner.state.strategies)

    if runner.mode == "REPLAY":
        mode_str = f"[bold bright_magenta]REPLAY[/bold bright_magenta]"
        detail_str = f"Ticks: {runner._total_ticks}  |  Speed: {sleep_seconds or 1.0}s/tick"
    else:
        mode_str = f"[bold cyan]LIVE[/bold cyan]"
        detail_str = (
            f"Duration: {runner.lcfg.duration_hours}h  |  "
            f"Interval: {runner.lcfg.interval_minutes}m"
        )

    console.print(
        Panel(
            f"[bold cyan]ETHU Neural Trader[/bold cyan]  -  {mode_str}\n"
            f"{detail_str}  |  "
            f"Capital: ${runner.lcfg.initial_capital:,.0f}\n"
            f"Ensemble: {n_models} models  |  "
            f"Strategies: {n_strats}  |  "
            f"Device: {DEVICE}",
            border_style="bright_blue",
        )
    )

    import time as _time

    if sleep_seconds is None:
        sleep_seconds = 1.0 if runner.mode == "REPLAY" else runner.lcfg.interval_minutes * 60

    with Live(build_display(runner), console=console, refresh_per_second=2, screen=True) as live:
        while not runner.is_done:
            runner.step()
            live.update(build_display(runner))
            _time.sleep(sleep_seconds)

    runner.print_final_summary()
    runner.save_results()
