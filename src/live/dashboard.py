"""Rich + plotext terminal dashboard for multi-ticker live and replay trading."""

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

from config import DEVICE, TARGET_TICKERS

if TYPE_CHECKING:
    from src.live.runner import LiveRunner, ReplayRunner


STRAT_COLORS = {
    "Conservative": "bright_red",
    "Moderate":     "bright_yellow",
    "Default":      "bright_green",
    "Aggressive":   "bright_cyan",
    "Ultra-Aggr":   "bright_magenta",
}

TICKER_COLORS = {
    "UVXY": "bright_red",
    "SPXU": "bright_yellow",
    "SVIX": "bright_cyan",
    "SPXL": "bright_green",
}

PLOT_TICKER_COLORS = {
    "UVXY": "red",
    "SPXU": "yellow",
    "SVIX": "cyan",
    "SPXL": "green",
}


def _build_chart(runner, width: int = 110, height: int = 18) -> str:
    """Build normalised performance chart for all tickers."""
    st = runner.state
    has_data = any(len(st.per_ticker_prices.get(t, [])) >= 2 for t in runner.tickers)
    if not has_data:
        return "  Waiting for data points..."

    plt.clf()
    plt.plotsize(width, height)
    plt.theme("dark")

    max_len = max(len(st.per_ticker_prices.get(t, [])) for t in runner.tickers)
    x = list(range(max_len))

    for tkr in runner.tickers:
        prices = st.per_ticker_prices.get(tkr, [])
        if len(prices) < 2:
            continue
        p0 = prices[0] if prices[0] != 0 else 1
        norm = [p / p0 * 100 for p in prices]
        tx = list(range(len(norm)))
        color = PLOT_TICKER_COLORS.get(tkr, "white")
        plt.plot(tx, norm, label=tkr, color=color)

    for name, ss in st.strategies.items():
        if name != "Default":
            continue
        if len(ss.portfolio_values) < 2:
            continue
        v0 = ss.portfolio_values[0] if ss.portfolio_values[0] != 0 else 1
        vals = [v / v0 * 100 for v in ss.portfolio_values]
        plt.plot(list(range(len(vals))), vals, label="Portfolio", color="white")

    plt.title("Normalised Performance (base=100)")
    plt.xlabel("Ticks")
    plt.ylabel("%")

    return plt.build()


def _build_ticker_panel(runner) -> Panel:
    """Show per-ticker prices, signals, and positions."""
    st = runner.state
    tbl = Table(show_header=True, border_style="bright_blue", padding=(0, 1))
    tbl.add_column("Ticker", style="bold", width=8)
    tbl.add_column("Price", justify="right", width=10)
    tbl.add_column("Return", justify="right", width=9)
    tbl.add_column("Signal", justify="right", width=10)
    tbl.add_column("Conf", justify="right", width=6)

    for tkr in runner.tickers:
        prices = st.per_ticker_prices.get(tkr, [])
        signals = st.per_ticker_signals.get(tkr, [])
        confs = st.per_ticker_confidences.get(tkr, [])

        if prices:
            price_str = f"${prices[-1]:.2f}"
            ret = st.price_return(tkr)
            ret_str = f"{ret:+.2%}"
            ret_style = "green" if ret >= 0 else "red"
        else:
            price_str = "---"
            ret_str = "---"
            ret_style = "dim"

        if signals:
            sig = signals[-1]
            if sig > 0.001:
                sig_style = "bold green"
            elif sig < -0.001:
                sig_style = "bold red"
            else:
                sig_style = "dim"
            sig_str = f"{sig:+.6f}"
        else:
            sig_str = "---"
            sig_style = "dim"

        conf_str = f"{confs[-1]:.0%}" if confs else "---"

        color = TICKER_COLORS.get(tkr, "white")
        tbl.add_row(
            Text(tkr, style=color),
            price_str,
            Text(ret_str, style=ret_style),
            Text(sig_str, style=sig_style),
            conf_str,
        )

    return Panel(tbl, title="[bright_white]Ticker Signals[/bright_white]", border_style="bright_blue")


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
    tbl.add_column("Positions", width=24)

    for name, ss in st.strategies.items():
        val = ss.portfolio_values[-1] if ss.portfolio_values else 0
        ret = ss.current_return
        ret_style = STRAT_COLORS.get(name, "white")
        wr = f"{ss.win_rate:.0%}" if ss.all_trade_returns else "---"
        dd = f"{ss.max_drawdown:+.1%}" if ss.max_drawdown < 0 else "0.0%"

        pos_parts = []
        for tkr, status in ss.active_positions.items():
            color = TICKER_COLORS.get(tkr, "white")
            style = f"bold {color}" if status == "LONG" else "dim"
            pos_parts.append(f"[{style}]{tkr}[/{style}]")
        pos_str = " ".join(pos_parts)

        tbl.add_row(
            Text(name, style=ret_style),
            f"${val:,.2f}",
            Text(f"{ret:+.2%}", style="green" if ret >= 0 else "red"),
            str(ss.total_trades),
            wr,
            Text(dd, style="red" if ss.max_drawdown < -0.02 else "dim"),
            f"{ss.sharpe:.2f}",
            Text.from_markup(pos_str),
        )

    return tbl


def _build_trade_log(runner, max_rows: int = 8) -> Panel:
    recs = [r for r in runner.state.records if r.action != "HOLD"]
    recent = recs[-max_rows:] if recs else []

    if not recent:
        return Panel("  No trades yet.", title="Recent Trades", border_style="blue")

    tbl = Table(show_header=True, border_style="dim", padding=(0, 1))
    tbl.add_column("Time", width=9)
    tbl.add_column("Ticker", width=6)
    tbl.add_column("Strategy", width=14)
    tbl.add_column("Action", width=6)
    tbl.add_column("Price", justify="right", width=10)
    tbl.add_column("Conf", justify="right", width=6)

    for r in recent:
        act_style = "bold green" if r.action == "BUY" else "bold red"
        ts_display = r.timestamp.split(" ")[-1][:8] if " " in r.timestamp else r.timestamp[:8]
        tkr_color = TICKER_COLORS.get(r.ticker, "white")
        tbl.add_row(
            ts_display,
            Text(r.ticker, style=tkr_color),
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
    n_models = runner.multi_predictor.total_models
    n_tickers = len(runner.tickers)
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

    header.append(f"    {model_name} x{n_models} ({n_tickers} tickers)", style="bold cyan")
    header.append(f"    {gpu}", style="bold yellow")
    header.append(f"    fetches: {fetches}", style="dim")
    return header


def build_display(runner) -> Panel:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="chart", size=20),
        Layout(name="middle", size=12),
        Layout(name="bottom", size=12),
    )

    layout["header"].update(
        Panel(_build_header(runner), style="bold", border_style="bright_blue")
    )

    chart_str = _build_chart(runner)
    layout["chart"].update(
        Panel(chart_str, title="[cyan]Multi-Ticker Performance[/cyan]", border_style="blue")
    )

    layout["middle"].update(_build_strategy_table(runner))

    layout["bottom"].split_row(
        Layout(name="tickers", ratio=1),
        Layout(name="trades", ratio=1),
    )
    layout["bottom"]["tickers"].update(_build_ticker_panel(runner))
    layout["bottom"]["trades"].update(_build_trade_log(runner))

    mode_label = "REPLAY" if runner.mode == "REPLAY" else "LIVE"
    tickers_str = " | ".join(runner.tickers)
    return Panel(
        layout,
        title=f"[bold bright_white] Multi-Ticker Neural Trader  -  {tickers_str}  -  {mode_label} [/bold bright_white]",
        border_style="bright_blue",
        padding=(0, 1),
    )


def run_dashboard(runner, sleep_seconds: float | None = None) -> None:
    """Main loop for both live and replay modes."""
    console = Console()

    n_models = runner.multi_predictor.total_models
    n_strats = len(runner.state.strategies)
    n_tickers = len(runner.tickers)

    instant = runner.mode == "REPLAY" and sleep_seconds is not None and sleep_seconds <= 0

    if runner.mode == "REPLAY":
        mode_str = f"[bold bright_magenta]REPLAY[/bold bright_magenta]"
        if instant:
            detail_str = f"Ticks: {runner._total_ticks}  |  Speed: instant"
        else:
            detail_str = f"Ticks: {runner._total_ticks}  |  Speed: {sleep_seconds or 1.0}s/tick"
    else:
        mode_str = f"[bold cyan]LIVE[/bold cyan]"
        detail_str = (
            f"Duration: {runner.lcfg.duration_hours}h  |  "
            f"Interval: {runner.lcfg.interval_minutes}m"
        )

    tickers_str = ", ".join(runner.tickers)
    console.print(
        Panel(
            f"[bold cyan]Multi-Ticker Neural Trader[/bold cyan]  -  {mode_str}\n"
            f"Tickers: {tickers_str}\n"
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

    if instant:
        total = runner._total_ticks
        last_pct = -1
        while not runner.is_done:
            runner.step()
            pct = int(runner._tick_idx / total * 100) if total > 0 else 0
            if pct >= last_pct + 10:
                console.print(f"  [dim]Processing... {runner._tick_idx}/{total} ({pct}%)[/dim]")
                last_pct = pct
        console.print()
        console.print(build_display(runner))
    else:
        with Live(build_display(runner), console=console, refresh_per_second=2, screen=True) as live:
            while not runner.is_done:
                runner.step()
                live.update(build_display(runner))
                _time.sleep(sleep_seconds)

    runner.print_final_summary()
    runner.save_results()
