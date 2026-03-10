"""Live and replay simulation with multi-ticker, multi-strategy portfolios.

Each of the 4 target tickers (UVXY, SPXU, SVIX, SPXL) gets independent
position management within each strategy. The multi-predictor provides
per-ticker ensemble predictions.
"""

from __future__ import annotations

import signal
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from config import (
    DEVICE, TARGET_TICKERS, ALL_TICKERS, LiveConfig, ModelConfig, PROCESSED_DIR,
)
from src.data.fetcher import fetch_intraday
from src.data.preprocessor import (
    build_features, load_scaler, load_feature_cols,
)
from src.model.predictor import MultiPredictor


LIVE_STRATEGIES = [
    {"name": "Conservative",  "long": 0.005,  "flat": -0.005},
    {"name": "Moderate",      "long": 0.002,  "flat": -0.002},
    {"name": "Default",       "long": 0.001,  "flat": -0.001},
    {"name": "Aggressive",    "long": 0.0005, "flat": -0.0005},
    {"name": "Ultra-Aggr",    "long": 0.0,    "flat": 0.0},
]


class SignalNormalizer:
    """Rolling z-score normalization that removes systematic model bias.

    Maintains independent normalization state per ticker.
    """

    def __init__(self, tickers: List[str], warmup: int = 30, window: int = 100):
        self.warmup = warmup
        self.window = window
        self._history: Dict[str, List[float]] = {t: [] for t in tickers}

    def update(self, ticker: str, raw: float) -> float:
        self._history[ticker].append(raw)
        hist = self._history[ticker]
        if len(hist) < self.warmup:
            return 0.0
        recent = hist[-self.window:]
        mu = np.mean(recent)
        sigma = np.std(recent)
        if sigma < 1e-10:
            return 0.0
        z = (raw - mu) / sigma
        return float(z * 0.003)


@dataclass
class TradeRecord:
    timestamp: str
    ticker: str
    strategy: str
    action: str
    price: float
    signal: float
    confidence: float
    agreement: float
    portfolio_value: float
    position: str


@dataclass
class TickerPosition:
    """Per-ticker position within a strategy."""
    ticker: str
    position: int = 0
    entry_price: float = 0.0
    num_trades: int = 0
    trade_returns: List[float] = field(default_factory=list)


@dataclass
class StrategyState:
    """Per-strategy portfolio tracking across all tickers."""
    name: str
    long_threshold: float
    flat_threshold: float
    capital: float
    ticker_positions: Dict[str, TickerPosition] = field(default_factory=dict)
    portfolio_values: List[float] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)

    @property
    def current_return(self) -> float:
        if len(self.portfolio_values) < 2:
            return 0.0
        return (self.portfolio_values[-1] / self.portfolio_values[0]) - 1

    @property
    def max_drawdown(self) -> float:
        if len(self.portfolio_values) < 2:
            return 0.0
        arr = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(arr)
        dd = (arr - peak) / np.where(peak > 0, peak, 1)
        return float(dd.min())

    @property
    def total_trades(self) -> int:
        return sum(tp.num_trades for tp in self.ticker_positions.values())

    @property
    def all_trade_returns(self) -> List[float]:
        rets = []
        for tp in self.ticker_positions.values():
            rets.extend(tp.trade_returns)
        return rets

    @property
    def win_rate(self) -> float:
        rets = self.all_trade_returns
        if not rets:
            return 0.0
        return sum(1 for r in rets if r > 0) / len(rets)

    @property
    def sharpe(self) -> float:
        if len(self.portfolio_values) < 3:
            return 0.0
        rets = np.diff(self.portfolio_values) / np.array(self.portfolio_values[:-1])
        if rets.std() == 0:
            return 0.0
        return float(rets.mean() / rets.std() * np.sqrt(252 * 78))

    @property
    def active_positions(self) -> Dict[str, str]:
        """Return {ticker: 'LONG'/'FLAT'} for all tickers."""
        return {
            tkr: ("LONG" if tp.position == 1 else "FLAT")
            for tkr, tp in self.ticker_positions.items()
        }


@dataclass
class LiveState:
    """Global state shared across all strategies."""
    strategies: Dict[str, StrategyState] = field(default_factory=dict)
    per_ticker_prices: Dict[str, List[float]] = field(default_factory=dict)
    per_ticker_signals: Dict[str, List[float]] = field(default_factory=dict)
    per_ticker_confidences: Dict[str, List[float]] = field(default_factory=dict)
    per_ticker_agreements: Dict[str, List[float]] = field(default_factory=dict)
    timestamps: List[str] = field(default_factory=list)
    records: List[TradeRecord] = field(default_factory=list)
    fetch_errors: int = 0
    fetch_successes: int = 0

    def price_return(self, ticker: str) -> float:
        prices = self.per_ticker_prices.get(ticker, [])
        if len(prices) < 2:
            return 0.0
        return (prices[-1] / prices[0]) - 1


def _init_strategies(strats, initial_capital, tickers):
    """Create a dict of StrategyState with per-ticker positions."""
    d = {}
    for s in strats:
        ss = StrategyState(
            name=s["name"],
            long_threshold=s["long"],
            flat_threshold=s["flat"],
            capital=initial_capital,
        )
        for tkr in tickers:
            ss.ticker_positions[tkr] = TickerPosition(ticker=tkr)
        ss.portfolio_values.append(initial_capital)
        d[s["name"]] = ss
    return d


def _step_strategy(
    ss: StrategyState,
    ticker_predictions: Dict[str, float],
    ticker_prices: Dict[str, float],
    prev_prices: Dict[str, Optional[float]],
    timestamp: str,
    ticker_details: Dict[str, dict],
    records: List[TradeRecord],
):
    """Advance a strategy's portfolio for all tickers in one step."""
    capital = ss.portfolio_values[-1]

    n_tickers = len(ss.ticker_positions)
    per_ticker_alloc = capital / max(n_tickers, 1)

    total_pnl = 0.0
    for tkr, tp in ss.ticker_positions.items():
        if tkr not in ticker_prices:
            continue
        price = ticker_prices[tkr]
        prev = prev_prices.get(tkr)
        if tp.position == 1 and prev and prev > 0:
            total_pnl += per_ticker_alloc * (price / prev - 1)

    capital += total_pnl

    action_summary = []
    for tkr, tp in ss.ticker_positions.items():
        if tkr not in ticker_predictions or tkr not in ticker_prices:
            continue

        pred = ticker_predictions[tkr]
        price = ticker_prices[tkr]
        detail = ticker_details.get(tkr, {"confidence": 0, "agreement": 0})
        action = "HOLD"

        if pred > ss.long_threshold and tp.position == 0:
            action = "BUY"
            tp.position = 1
            tp.entry_price = price
            tp.num_trades += 1
        elif pred < ss.flat_threshold and tp.position == 1:
            action = "SELL"
            trade_ret = (price / tp.entry_price - 1) if tp.entry_price > 0 else 0.0
            tp.trade_returns.append(trade_ret)
            tp.position = 0
            tp.entry_price = 0.0
            tp.num_trades += 1

        if action != "HOLD":
            action_summary.append(f"{tkr}:{action}")
            records.append(TradeRecord(
                timestamp=timestamp, ticker=tkr, strategy=ss.name, action=action,
                price=price, signal=pred, confidence=detail.get("confidence", 0),
                agreement=detail.get("agreement", 0), portfolio_value=capital,
                position="LONG" if tp.position == 1 else "FLAT",
            ))

    ss.capital = capital
    ss.portfolio_values.append(capital)
    ss.actions.append("|".join(action_summary) if action_summary else "HOLD")


class LiveRunner:
    """Multi-ticker, multi-strategy live simulation."""

    mode = "LIVE"

    def __init__(
        self,
        multi_predictor: MultiPredictor,
        live_cfg: LiveConfig | None = None,
        model_cfg: ModelConfig | None = None,
        strategies: List[dict] | None = None,
    ):
        self.multi_predictor = multi_predictor
        self.lcfg = live_cfg or LiveConfig()
        self.mcfg = model_cfg or ModelConfig()
        self.tickers = list(multi_predictor.predictors.keys())
        self._stop = False

        self._scalers: Dict[str, object] = {}
        self._saved_feature_cols: Dict[str, Optional[List[str]]] = {}
        for tkr in self.tickers:
            self._scalers[tkr] = load_scaler(ticker=tkr)
            try:
                self._saved_feature_cols[tkr] = load_feature_cols(ticker=tkr)
            except FileNotFoundError:
                self._saved_feature_cols[tkr] = None

        self._normalizer = SignalNormalizer(self.tickers, warmup=30, window=100)

        strats = strategies or LIVE_STRATEGIES
        self.state = LiveState()
        for tkr in self.tickers:
            self.state.per_ticker_prices[tkr] = []
            self.state.per_ticker_signals[tkr] = []
            self.state.per_ticker_confidences[tkr] = []
            self.state.per_ticker_agreements[tkr] = []
        self.state.strategies = _init_strategies(strats, self.lcfg.initial_capital, self.tickers)

        signal.signal(signal.SIGINT, self._handle_stop)

    def _handle_stop(self, *_):
        self._stop = True

    @property
    def start_time(self) -> dt.datetime:
        if not hasattr(self, "_start_time"):
            self._start_time = dt.datetime.now()
        return self._start_time

    @property
    def elapsed(self) -> dt.timedelta:
        return dt.datetime.now() - self.start_time

    @property
    def remaining(self) -> dt.timedelta:
        total = dt.timedelta(hours=self.lcfg.duration_hours)
        rem = total - self.elapsed
        return max(rem, dt.timedelta(0))

    @property
    def is_done(self) -> bool:
        return self._stop or self.elapsed.total_seconds() >= self.lcfg.duration_hours * 3600

    @property
    def tick_progress(self) -> str:
        return ""

    def _fetch_features(self) -> Dict[str, Optional[np.ndarray]]:
        """Fetch intraday data and build feature windows for all tickers."""
        result: Dict[str, Optional[np.ndarray]] = {t: None for t in self.tickers}
        try:
            data = fetch_intraday(
                tickers=ALL_TICKERS,
                interval=self.lcfg.intraday_interval,
                period="5d",
            )
            if not data:
                return result

            merged = pd.concat(data, axis=1)
            merged.sort_index(inplace=True)

            for tkr in self.tickers:
                if tkr not in data:
                    continue
                try:
                    feat_df, target_col = build_features(
                        merged, target_ticker=tkr, quiet=True, prune=False,
                    )
                    if len(feat_df) < self.mcfg.seq_len:
                        continue

                    saved_cols = self._saved_feature_cols.get(tkr)
                    if saved_cols is not None:
                        available = [c for c in feat_df.columns if c != target_col]
                        for col in saved_cols:
                            if col not in available:
                                feat_df[col] = 0.0
                        X = feat_df[saved_cols].values.astype(np.float32)
                    else:
                        feature_cols = [c for c in feat_df.columns if c != target_col]
                        X = feat_df[feature_cols].values.astype(np.float32)

                    if len(X) == 0:
                        continue

                    scaler = self._scalers[tkr]
                    X_scaled = scaler.transform(X)
                    result[tkr] = X_scaled[-self.mcfg.seq_len:]
                except Exception:
                    continue

        except Exception:
            pass

        return result

    def _get_current_prices(self) -> Dict[str, Optional[float]]:
        """Get latest prices for all target tickers."""
        prices: Dict[str, Optional[float]] = {}
        try:
            data = fetch_intraday(
                tickers=self.tickers,
                interval="1m",
                period="1d",
            )
            for tkr in self.tickers:
                if tkr in data and not data[tkr].empty:
                    prices[tkr] = float(data[tkr]["close"].iloc[-1])
                else:
                    prices[tkr] = None
        except Exception:
            for tkr in self.tickers:
                prices[tkr] = None
        return prices

    def step(self) -> bool:
        windows = self._fetch_features()
        current_prices = self._get_current_prices()

        any_valid = False
        ticker_predictions: Dict[str, float] = {}
        ticker_details: Dict[str, dict] = {}
        valid_prices: Dict[str, float] = {}
        prev_prices: Dict[str, Optional[float]] = {}

        for tkr in self.tickers:
            w = windows.get(tkr)
            p = current_prices.get(tkr)
            if w is None or p is None:
                continue

            any_valid = True
            detail = self.multi_predictor.predict_detailed(tkr, w)
            raw_pred = detail["mean"]
            pred = self._normalizer.update(tkr, raw_pred)

            ticker_predictions[tkr] = pred
            ticker_details[tkr] = detail
            valid_prices[tkr] = p
            prev_list = self.state.per_ticker_prices.get(tkr, [])
            prev_prices[tkr] = prev_list[-1] if prev_list else None

        if not any_valid:
            self.state.fetch_errors += 1
            return False

        self.state.fetch_successes += 1
        now = dt.datetime.now().strftime("%H:%M:%S")
        self.state.timestamps.append(now)

        for tkr in self.tickers:
            if tkr in valid_prices:
                self.state.per_ticker_prices[tkr].append(valid_prices[tkr])
                self.state.per_ticker_signals[tkr].append(ticker_predictions.get(tkr, 0.0))
                d = ticker_details.get(tkr, {})
                self.state.per_ticker_confidences[tkr].append(d.get("confidence", 0))
                self.state.per_ticker_agreements[tkr].append(d.get("agreement", 0))

        for ss in self.state.strategies.values():
            _step_strategy(ss, ticker_predictions, valid_prices, prev_prices, now, ticker_details, self.state.records)

        return True

    def save_results(self, path: str = "live_results.csv") -> None:
        if not self.state.records:
            return
        rows = [
            {
                "timestamp": r.timestamp, "ticker": r.ticker, "strategy": r.strategy,
                "action": r.action, "price": r.price, "signal": r.signal,
                "confidence": r.confidence, "agreement": r.agreement,
                "portfolio_value": r.portfolio_value, "position": r.position,
            }
            for r in self.state.records
        ]
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"\n[LIVE] Results saved to {path}")

    def print_final_summary(self):
        st = self.state
        print(f"\n{'='*90}")
        print(f"  SESSION SUMMARY  ({self.mode})")
        print(f"{'='*90}")
        print(f"  Duration:       {self.elapsed}")
        print(f"  Data points:    {st.fetch_successes} ok, {st.fetch_errors} failed")
        print(f"  Ensemble:       {self.multi_predictor.total_models} models total")

        for tkr in self.tickers:
            prices = st.per_ticker_prices.get(tkr, [])
            if prices:
                print(f"  {tkr} price:     ${prices[0]:.2f} -> ${prices[-1]:.2f} ({st.price_return(tkr):+.2%})")

        print(f"\n  {'Strategy':<16} {'Return':>9} {'Value':>12} {'Trades':>7} {'WinRate':>8} {'MaxDD':>8} {'Sharpe':>7} {'Positions':>20}")
        print(f"  {'-'*87}")
        for ss in st.strategies.values():
            pos_str = " ".join(f"{t}:{p}" for t, p in ss.active_positions.items())
            val = ss.portfolio_values[-1] if ss.portfolio_values else 0
            print(
                f"  {ss.name:<16} {ss.current_return:>+8.2%} ${val:>10,.2f} "
                f"{ss.total_trades:>7d} {ss.win_rate:>7.1%} {ss.max_drawdown:>+7.2%} "
                f"{ss.sharpe:>7.2f} {pos_str:>20}"
            )
        print(f"  {'-'*87}")
        for tkr in self.tickers:
            ret = st.price_return(tkr)
            print(f"  {'B&H ' + tkr:<16} {ret:>+8.2%}")
        print(f"{'='*90}")


# ---------------------------------------------------------------------------
# Historical replay runner
# ---------------------------------------------------------------------------

INTERVAL_LIMITS = {
    "1m": ("7d", "7 days"),
    "2m": ("60d", "60 days"),
    "5m": ("60d", "60 days"),
    "15m": ("60d", "60 days"),
    "30m": ("60d", "60 days"),
    "1h": ("730d", "~2 years"),
}


class ReplayRunner:
    """Replays historical intraday data through the multi-ticker engine."""

    mode = "REPLAY"

    def __init__(
        self,
        multi_predictor: MultiPredictor,
        live_cfg: LiveConfig | None = None,
        model_cfg: ModelConfig | None = None,
        strategies: List[dict] | None = None,
        replay_hours: float = 24.0,
        replay_interval: str = "1m",
    ):
        self.multi_predictor = multi_predictor
        self.lcfg = live_cfg or LiveConfig()
        self.mcfg = model_cfg or ModelConfig()
        self.tickers = list(multi_predictor.predictors.keys())
        self._stop = False
        self.replay_interval = replay_interval

        self._scalers: Dict[str, object] = {}
        self._saved_feature_cols: Dict[str, Optional[List[str]]] = {}
        for tkr in self.tickers:
            self._scalers[tkr] = load_scaler(ticker=tkr)
            try:
                self._saved_feature_cols[tkr] = load_feature_cols(ticker=tkr)
            except FileNotFoundError:
                self._saved_feature_cols[tkr] = None

        self._normalizer = SignalNormalizer(self.tickers, warmup=30, window=100)

        strats = strategies or LIVE_STRATEGIES
        self.state = LiveState()
        for tkr in self.tickers:
            self.state.per_ticker_prices[tkr] = []
            self.state.per_ticker_signals[tkr] = []
            self.state.per_ticker_confidences[tkr] = []
            self.state.per_ticker_agreements[tkr] = []
        self.state.strategies = _init_strategies(strats, self.lcfg.initial_capital, self.tickers)

        self._tick_idx = 0
        self._per_ticker_windows: Dict[str, List[np.ndarray]] = {t: [] for t in self.tickers}
        self._per_ticker_prices: Dict[str, List[float]] = {t: [] for t in self.tickers}
        self._timestamps: List[str] = []
        self._total_ticks = 0

        self._load_replay_data(replay_hours, replay_interval)

        signal.signal(signal.SIGINT, self._handle_stop)

    def _handle_stop(self, *_):
        self._stop = True

    def _load_replay_data(self, hours: float, interval: str):
        """Fetch historical intraday data and pre-compute all feature windows."""
        limit_period, limit_desc = INTERVAL_LIMITS.get(interval, ("7d", "7 days"))

        if interval == "1m":
            period = f"{min(int(hours / 24) + 2, 7)}d"
        elif interval in ("2m", "5m", "15m", "30m"):
            period = f"{min(int(hours / 24) + 2, 60)}d"
        else:
            period = f"{min(int(hours / 24) + 2, 730)}d"

        print(f"[REPLAY] Fetching {interval} data (period={period}, max={limit_desc})...")
        data = fetch_intraday(tickers=ALL_TICKERS, interval=interval, period=period)

        if not data:
            raise RuntimeError("Failed to fetch historical data for replay.")

        merged = pd.concat(data, axis=1)
        merged.sort_index(inplace=True)

        if hours > 0:
            cutoff = merged.index[-1] - pd.Timedelta(hours=hours)
            replay_start = merged.index.searchsorted(cutoff)
        else:
            replay_start = 0

        max_ticks = 0
        for tkr in self.tickers:
            if tkr not in data:
                print(f"[REPLAY] WARNING: No data for {tkr}")
                continue

            target_close = merged.get((tkr, "close"))
            if target_close is None or target_close.dropna().empty:
                print(f"[REPLAY] WARNING: No {tkr} price data")
                continue

            try:
                feat_df, target_col = build_features(
                    merged, target_ticker=tkr, quiet=True, prune=False,
                )
            except Exception as e:
                print(f"[REPLAY] WARNING: Could not build features for {tkr}: {e}")
                continue

            if len(feat_df) == 0:
                print(f"[REPLAY] WARNING: Empty feature DataFrame for {tkr}, skipping")
                continue

            saved_cols = self._saved_feature_cols.get(tkr)
            if saved_cols is not None:
                available = set(c for c in feat_df.columns if c != target_col)
                for col in saved_cols:
                    if col not in available:
                        feat_df[col] = 0.0
                X = feat_df[saved_cols].values.astype(np.float32)
                n_matched = sum(1 for c in saved_cols if c in available)
                print(f"[REPLAY] [{tkr}] Feature match: {n_matched}/{len(saved_cols)}")
            else:
                feature_cols = [c for c in feat_df.columns if c != target_col]
                X = feat_df[feature_cols].values.astype(np.float32)

            if len(X) == 0:
                print(f"[REPLAY] WARNING: No valid samples for {tkr} after feature selection, skipping")
                continue

            scaler = self._scalers[tkr]
            X_scaled = scaler.transform(X)

            ticker_close = merged[(tkr, "close")].reindex(feat_df.index)
            ts_index = feat_df.index

            replay_feat_start = max(0, feat_df.index.searchsorted(merged.index[replay_start]) if replay_start > 0 else 0)

            seq = self.mcfg.seq_len
            ticker_windows = []
            ticker_prices = []
            ticker_timestamps = []

            for i in range(max(seq, replay_feat_start), len(X_scaled)):
                w = X_scaled[i - seq : i]
                p = ticker_close.iloc[i]
                if np.isnan(p) or p <= 0:
                    continue
                ts = str(ts_index[i])
                ticker_windows.append(w)
                ticker_prices.append(float(p))
                ticker_timestamps.append(ts)

            self._per_ticker_windows[tkr] = ticker_windows
            self._per_ticker_prices[tkr] = ticker_prices

            if len(ticker_timestamps) > len(self._timestamps):
                self._timestamps = ticker_timestamps

            max_ticks = max(max_ticks, len(ticker_windows))
            print(f"[REPLAY] [{tkr}] Loaded {len(ticker_windows)} ticks")

        self._total_ticks = max_ticks
        print(f"[REPLAY] Total: {self._total_ticks} ticks ({interval} over ~{hours:.0f}h)")

        if self._total_ticks == 0:
            raise RuntimeError("No valid ticks in replay window.")

    @property
    def start_time(self) -> dt.datetime:
        if not hasattr(self, "_start_time"):
            self._start_time = dt.datetime.now()
        return self._start_time

    @property
    def elapsed(self) -> dt.timedelta:
        return dt.datetime.now() - self.start_time

    @property
    def remaining(self) -> dt.timedelta:
        return dt.timedelta(0)

    @property
    def is_done(self) -> bool:
        return self._stop or self._tick_idx >= self._total_ticks

    @property
    def tick_progress(self) -> str:
        return f"{self._tick_idx}/{self._total_ticks}"

    def step(self) -> bool:
        if self._tick_idx >= self._total_ticks:
            return False

        idx = self._tick_idx
        self._tick_idx += 1

        timestamp = self._timestamps[idx] if idx < len(self._timestamps) else str(idx)

        ticker_predictions: Dict[str, float] = {}
        ticker_details: Dict[str, dict] = {}
        valid_prices: Dict[str, float] = {}
        prev_prices: Dict[str, Optional[float]] = {}

        any_valid = False
        for tkr in self.tickers:
            windows = self._per_ticker_windows.get(tkr, [])
            prices = self._per_ticker_prices.get(tkr, [])
            if idx >= len(windows) or idx >= len(prices):
                continue

            window = windows[idx]
            price = prices[idx]
            any_valid = True

            detail = self.multi_predictor.predict_detailed(tkr, window)
            raw_pred = detail["mean"]
            pred = self._normalizer.update(tkr, raw_pred)

            ticker_predictions[tkr] = pred
            ticker_details[tkr] = detail
            valid_prices[tkr] = price
            prev_list = self.state.per_ticker_prices.get(tkr, [])
            prev_prices[tkr] = prev_list[-1] if prev_list else None

        if not any_valid:
            return False

        self.state.fetch_successes += 1
        self.state.timestamps.append(timestamp)

        for tkr in self.tickers:
            if tkr in valid_prices:
                self.state.per_ticker_prices[tkr].append(valid_prices[tkr])
                self.state.per_ticker_signals[tkr].append(ticker_predictions.get(tkr, 0.0))
                d = ticker_details.get(tkr, {})
                self.state.per_ticker_confidences[tkr].append(d.get("confidence", 0))
                self.state.per_ticker_agreements[tkr].append(d.get("agreement", 0))

        for ss in self.state.strategies.values():
            _step_strategy(ss, ticker_predictions, valid_prices, prev_prices, timestamp, ticker_details, self.state.records)

        return True

    def save_results(self, path: str = "replay_results.csv") -> None:
        if not self.state.records:
            return
        rows = [
            {
                "timestamp": r.timestamp, "ticker": r.ticker, "strategy": r.strategy,
                "action": r.action, "price": r.price, "signal": r.signal,
                "confidence": r.confidence, "agreement": r.agreement,
                "portfolio_value": r.portfolio_value, "position": r.position,
            }
            for r in self.state.records
        ]
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"\n[REPLAY] Results saved to {path}")

    def print_final_summary(self):
        st = self.state
        print(f"\n{'='*90}")
        print(f"  REPLAY SUMMARY")
        print(f"{'='*90}")
        processed = self._tick_idx
        total = self._total_ticks
        pct = processed / total * 100 if total > 0 else 0
        if processed < total:
            print(f"  Replayed:       {processed}/{total} ticks ({pct:.0f}%) ({self.replay_interval})")
        else:
            print(f"  Replayed:       {total} ticks ({self.replay_interval})")
        print(f"  Wall time:      {self.elapsed}")
        print(f"  Ensemble:       {self.multi_predictor.total_models} models total")

        for tkr in self.tickers:
            prices = st.per_ticker_prices.get(tkr, [])
            if prices:
                print(f"  {tkr} price:     ${prices[0]:.2f} -> ${prices[-1]:.2f} ({st.price_return(tkr):+.2%})")

        if st.timestamps:
            print(f"  Time range:     {st.timestamps[0]} -> {st.timestamps[-1]}")

        print(f"\n  {'Strategy':<16} {'Return':>9} {'Value':>12} {'Trades':>7} {'WinRate':>8} {'MaxDD':>8} {'Sharpe':>7}")
        print(f"  {'-'*67}")
        for ss in st.strategies.values():
            val = ss.portfolio_values[-1] if ss.portfolio_values else 0
            print(
                f"  {ss.name:<16} {ss.current_return:>+8.2%} ${val:>10,.2f} "
                f"{ss.total_trades:>7d} {ss.win_rate:>7.1%} {ss.max_drawdown:>+7.2%} "
                f"{ss.sharpe:>7.2f}"
            )
        print(f"  {'-'*67}")
        for tkr in self.tickers:
            ret = st.price_return(tkr)
            print(f"  {'B&H ' + tkr:<16} {ret:>+8.2%}")
        print(f"{'='*90}")
