"""Live and replay simulation with multi-strategy portfolios and ensemble confidence."""

from __future__ import annotations

import signal
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from config import (
    DEVICE, TARGET_TICKER, ALL_TICKERS, LiveConfig, ModelConfig, PROCESSED_DIR,
)
from src.data.fetcher import fetch_intraday
from src.data.preprocessor import (
    build_features, load_scaler, load_feature_cols,
)
from src.model.predictor import Predictor


LIVE_STRATEGIES = [
    {"name": "Conservative",  "long": 0.005,  "flat": -0.005},
    {"name": "Moderate",      "long": 0.002,  "flat": -0.002},
    {"name": "Default",       "long": 0.001,  "flat": -0.001},
    {"name": "Aggressive",    "long": 0.0005, "flat": -0.0005},
    {"name": "Ultra-Aggr",    "long": 0.0,    "flat": 0.0},
]


class SignalNormalizer:
    """Rolling z-score normalization that removes systematic model bias.

    The model trained on daily data produces predictions on a daily-return scale.
    When fed intraday data, predictions cluster far from zero. This normalizer
    tracks recent predictions and converts the raw value to how many standard
    deviations it deviates from the recent mean, yielding a zero-centered signal
    that works with the strategy thresholds.
    """

    def __init__(self, warmup: int = 30, window: int = 100):
        self.warmup = warmup
        self.window = window
        self._history: List[float] = []

    def update(self, raw: float) -> float:
        self._history.append(raw)
        if len(self._history) < self.warmup:
            return 0.0
        recent = self._history[-self.window:]
        mu = np.mean(recent)
        sigma = np.std(recent)
        if sigma < 1e-10:
            return 0.0
        z = (raw - mu) / sigma
        return float(z * 0.003)


@dataclass
class TradeRecord:
    timestamp: str
    strategy: str
    action: str
    price: float
    signal: float
    confidence: float
    agreement: float
    portfolio_value: float
    position: str


@dataclass
class StrategyState:
    """Per-strategy portfolio tracking."""
    name: str
    long_threshold: float
    flat_threshold: float
    capital: float
    position: int = 0
    entry_price: float = 0.0
    num_trades: int = 0
    trade_returns: List[float] = field(default_factory=list)
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
    def win_rate(self) -> float:
        if not self.trade_returns:
            return 0.0
        return sum(1 for r in self.trade_returns if r > 0) / len(self.trade_returns)

    @property
    def sharpe(self) -> float:
        if len(self.portfolio_values) < 3:
            return 0.0
        rets = np.diff(self.portfolio_values) / np.array(self.portfolio_values[:-1])
        if rets.std() == 0:
            return 0.0
        return float(rets.mean() / rets.std() * np.sqrt(252 * 78))


@dataclass
class LiveState:
    """Global state shared across all strategies."""
    strategies: Dict[str, StrategyState] = field(default_factory=dict)
    prices: List[float] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)
    signals: List[float] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    agreements: List[float] = field(default_factory=list)
    pred_stds: List[float] = field(default_factory=list)
    per_model_preds: List[List[float]] = field(default_factory=list)
    records: List[TradeRecord] = field(default_factory=list)
    fetch_errors: int = 0
    fetch_successes: int = 0

    @property
    def price_return(self) -> float:
        if len(self.prices) < 2:
            return 0.0
        return (self.prices[-1] / self.prices[0]) - 1


def _init_strategies(strats, initial_capital):
    """Create a dict of StrategyState from strategy config list."""
    d = {}
    for s in strats:
        ss = StrategyState(
            name=s["name"],
            long_threshold=s["long"],
            flat_threshold=s["flat"],
            capital=initial_capital,
        )
        ss.portfolio_values.append(initial_capital)
        d[s["name"]] = ss
    return d


def _step_strategy(ss, pred, price, prev_price, timestamp, detail, records):
    """Advance a single strategy's portfolio. Shared by live and replay."""
    if ss.position == 1 and prev_price and prev_price > 0:
        ss.capital = ss.portfolio_values[-1] * (price / prev_price)

    action = "HOLD"

    if pred > ss.long_threshold and ss.position == 0:
        action = "BUY"
        ss.position = 1
        ss.entry_price = price
        ss.num_trades += 1
    elif pred < ss.flat_threshold and ss.position == 1:
        action = "SELL"
        trade_ret = (price / ss.entry_price - 1) if ss.entry_price > 0 else 0.0
        ss.trade_returns.append(trade_ret)
        ss.position = 0
        ss.entry_price = 0.0
        ss.num_trades += 1

    ss.portfolio_values.append(ss.capital)
    ss.actions.append(action)

    records.append(TradeRecord(
        timestamp=timestamp, strategy=ss.name, action=action,
        price=price, signal=pred, confidence=detail["confidence"],
        agreement=detail["agreement"], portfolio_value=ss.capital,
        position="LONG" if ss.position == 1 else "FLAT",
    ))


class LiveRunner:
    """Multi-strategy live simulation with ensemble confidence tracking."""

    mode = "LIVE"

    def __init__(
        self,
        predictor: Predictor,
        live_cfg: LiveConfig | None = None,
        model_cfg: ModelConfig | None = None,
        strategies: List[dict] | None = None,
    ):
        self.predictor = predictor
        self.lcfg = live_cfg or LiveConfig()
        self.mcfg = model_cfg or ModelConfig()
        self.scaler = load_scaler()
        self.n_scaler_features = len(self.scaler.center_)
        self._stop = False

        try:
            self._saved_feature_cols = load_feature_cols()
        except FileNotFoundError:
            self._saved_feature_cols = None

        self._normalizer = SignalNormalizer(warmup=30, window=100)

        strats = strategies or LIVE_STRATEGIES
        self.state = LiveState()
        self.state.strategies = _init_strategies(strats, self.lcfg.initial_capital)

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

    def _fetch_features(self) -> Optional[np.ndarray]:
        try:
            data = fetch_intraday(
                tickers=ALL_TICKERS,
                interval=self.lcfg.intraday_interval,
                period="5d",
            )
            if not data or TARGET_TICKER not in data:
                return None

            merged = pd.concat(data, axis=1)
            merged.sort_index(inplace=True)

            feat_df, target_col = build_features(merged, quiet=True)

            if len(feat_df) < self.mcfg.seq_len:
                return None

            X, ok = self._select_saved_features(feat_df, target_col)
            if not ok:
                return None

            X_scaled = self.scaler.transform(X)
            return X_scaled[-self.mcfg.seq_len:]

        except Exception:
            return None

    def _select_saved_features(self, feat_df, target_col):
        """Select exactly the features the model was trained on, by name."""
        if self._saved_feature_cols is not None:
            available = [c for c in feat_df.columns if c != target_col]
            present = [c for c in self._saved_feature_cols if c in available]
            if len(present) < len(self._saved_feature_cols):
                missing = set(self._saved_feature_cols) - set(present)
                for col in missing:
                    feat_df[col] = 0.0
            X = feat_df[self._saved_feature_cols].values.astype(np.float32)
            return X, True

        feature_cols = [c for c in feat_df.columns if c != target_col]
        X = feat_df[feature_cols].values.astype(np.float32)
        if X.shape[1] != self.n_scaler_features:
            return X, False
        return X, True

    def get_current_price(self) -> Optional[float]:
        try:
            data = fetch_intraday(
                tickers=[TARGET_TICKER],
                interval="1m",
                period="1d",
            )
            if TARGET_TICKER in data and not data[TARGET_TICKER].empty:
                return float(data[TARGET_TICKER]["close"].iloc[-1])
        except Exception:
            pass
        return None

    def step(self) -> bool:
        window = self._fetch_features()
        price = self.get_current_price()

        if window is None or price is None:
            self.state.fetch_errors += 1
            return False

        self.state.fetch_successes += 1
        detail = self.predictor.predict_detailed(window)
        raw_pred = detail["mean"]
        pred = self._normalizer.update(raw_pred)
        now = dt.datetime.now().strftime("%H:%M:%S")
        prev_price = self.state.prices[-1] if self.state.prices else None

        self.state.prices.append(price)
        self.state.timestamps.append(now)
        self.state.signals.append(pred)
        self.state.confidences.append(detail["confidence"])
        self.state.agreements.append(detail["agreement"])
        self.state.pred_stds.append(detail["std"])
        self.state.per_model_preds.append(detail["individual"])

        for ss in self.state.strategies.values():
            _step_strategy(ss, pred, price, prev_price, now, detail, self.state.records)

        return True

    def save_results(self, path: str = "live_results.csv") -> None:
        if not self.state.records:
            return
        rows = [
            {
                "timestamp": r.timestamp, "strategy": r.strategy,
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
        print(f"\n{'='*80}")
        print(f"  SESSION SUMMARY  ({self.mode})")
        print(f"{'='*80}")
        print(f"  Duration:       {self.elapsed}")
        print(f"  Data points:    {st.fetch_successes} ok, {st.fetch_errors} failed")
        if st.prices:
            print(f"  ETHU price:     ${st.prices[0]:.2f} -> ${st.prices[-1]:.2f} ({st.price_return:+.2%})")
        print(f"  Ensemble size:  {len(self.predictor.models)} models")

        if st.signals:
            print(f"  Avg signal:     {np.mean(st.signals):+.6f}")
            print(f"  Avg confidence: {np.mean(st.confidences):.1%}")
            print(f"  Avg agreement:  {np.mean(st.agreements):.1%}")

        print(f"\n  {'Strategy':<16} {'Return':>9} {'Value':>12} {'Trades':>7} {'WinRate':>8} {'MaxDD':>8} {'Sharpe':>7} {'Pos':>6}")
        print(f"  {'-'*73}")
        for ss in st.strategies.values():
            pos = "LONG" if ss.position == 1 else "FLAT"
            val = ss.portfolio_values[-1] if ss.portfolio_values else 0
            print(
                f"  {ss.name:<16} {ss.current_return:>+8.2%} ${val:>10,.2f} "
                f"{ss.num_trades:>7d} {ss.win_rate:>7.1%} {ss.max_drawdown:>+7.2%} "
                f"{ss.sharpe:>7.2f} {pos:>6}"
            )
        print(f"  {'-'*73}")
        if st.prices:
            print(f"  {'Buy & Hold':<16} {st.price_return:>+8.2%}")
        print(f"{'='*80}")


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
    """Replays historical intraday data through the same multi-strategy engine.

    Pre-fetches all data up front, then steps through one tick at a time.
    """

    mode = "REPLAY"

    def __init__(
        self,
        predictor: Predictor,
        live_cfg: LiveConfig | None = None,
        model_cfg: ModelConfig | None = None,
        strategies: List[dict] | None = None,
        replay_hours: float = 24.0,
        replay_interval: str = "1m",
    ):
        self.predictor = predictor
        self.lcfg = live_cfg or LiveConfig()
        self.mcfg = model_cfg or ModelConfig()
        self.scaler = load_scaler()
        self.n_scaler_features = len(self.scaler.center_)
        self._stop = False
        self.replay_interval = replay_interval

        try:
            self._saved_feature_cols = load_feature_cols()
        except FileNotFoundError:
            self._saved_feature_cols = None

        self._normalizer = SignalNormalizer(warmup=30, window=100)

        strats = strategies or LIVE_STRATEGIES
        self.state = LiveState()
        self.state.strategies = _init_strategies(strats, self.lcfg.initial_capital)

        self._tick_idx = 0
        self._windows: List[np.ndarray] = []
        self._prices: List[float] = []
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

        if not data or TARGET_TICKER not in data:
            raise RuntimeError("Failed to fetch historical data for replay.")

        merged = pd.concat(data, axis=1)
        merged.sort_index(inplace=True)

        target_close = merged[(TARGET_TICKER, "close")] if (TARGET_TICKER, "close") in merged.columns else None
        if target_close is None or target_close.dropna().empty:
            raise RuntimeError(f"No {TARGET_TICKER} price data in fetched interval.")

        if hours > 0:
            cutoff = merged.index[-1] - pd.Timedelta(hours=hours)
            replay_start = merged.index.searchsorted(cutoff)
        else:
            replay_start = 0

        feat_df, target_col = build_features(merged, quiet=True)

        if self._saved_feature_cols is not None:
            available = set(c for c in feat_df.columns if c != target_col)
            for col in self._saved_feature_cols:
                if col not in available:
                    feat_df[col] = 0.0
            X = feat_df[self._saved_feature_cols].values.astype(np.float32)
            n_matched = sum(1 for c in self._saved_feature_cols if c in available)
            print(f"[REPLAY] Feature match: {n_matched}/{len(self._saved_feature_cols)} training features found")
        else:
            feature_cols = [c for c in feat_df.columns if c != target_col]
            X = feat_df[feature_cols].values.astype(np.float32)
            if X.shape[1] != self.n_scaler_features:
                raise RuntimeError(
                    f"Feature count mismatch: got {X.shape[1]}, scaler expects {self.n_scaler_features}. "
                    "Retrain the model to generate feature_cols.pkl."
                )

        X_scaled = self.scaler.transform(X)

        ethu_close = merged[(TARGET_TICKER, "close")].reindex(feat_df.index)
        ts_index = feat_df.index

        replay_feat_start = max(0, feat_df.index.searchsorted(merged.index[replay_start]) if replay_start > 0 else 0)

        seq = self.mcfg.seq_len
        for i in range(max(seq, replay_feat_start), len(X_scaled)):
            w = X_scaled[i - seq : i]
            p = ethu_close.iloc[i]
            if np.isnan(p) or p <= 0:
                continue
            ts = str(ts_index[i])
            self._windows.append(w)
            self._prices.append(float(p))
            self._timestamps.append(ts)

        self._total_ticks = len(self._windows)
        print(f"[REPLAY] Loaded {self._total_ticks} ticks ({interval} over ~{hours:.0f}h)")

        if self._total_ticks == 0:
            raise RuntimeError("No valid ticks in replay window. Try a larger --replay-hours.")

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

        window = self._windows[self._tick_idx]
        price = self._prices[self._tick_idx]
        timestamp = self._timestamps[self._tick_idx]
        self._tick_idx += 1

        detail = self.predictor.predict_detailed(window)
        raw_pred = detail["mean"]
        pred = self._normalizer.update(raw_pred)
        prev_price = self.state.prices[-1] if self.state.prices else None

        self.state.fetch_successes += 1
        self.state.prices.append(price)
        self.state.timestamps.append(timestamp)
        self.state.signals.append(pred)
        self.state.confidences.append(detail["confidence"])
        self.state.agreements.append(detail["agreement"])
        self.state.pred_stds.append(detail["std"])
        self.state.per_model_preds.append(detail["individual"])

        for ss in self.state.strategies.values():
            _step_strategy(ss, pred, price, prev_price, timestamp, detail, self.state.records)

        return True

    def save_results(self, path: str = "replay_results.csv") -> None:
        if not self.state.records:
            return
        rows = [
            {
                "timestamp": r.timestamp, "strategy": r.strategy,
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
        print(f"\n{'='*80}")
        print(f"  REPLAY SUMMARY")
        print(f"{'='*80}")
        print(f"  Replayed:       {self._total_ticks} ticks ({self.replay_interval})")
        print(f"  Wall time:      {self.elapsed}")
        if st.prices:
            print(f"  ETHU price:     ${st.prices[0]:.2f} -> ${st.prices[-1]:.2f} ({st.price_return:+.2%})")
            print(f"  Time range:     {st.timestamps[0]} -> {st.timestamps[-1]}")
        print(f"  Ensemble size:  {len(self.predictor.models)} models")

        if st.signals:
            print(f"  Avg signal:     {np.mean(st.signals):+.6f}")
            print(f"  Avg confidence: {np.mean(st.confidences):.1%}")
            print(f"  Avg agreement:  {np.mean(st.agreements):.1%}")

        print(f"\n  {'Strategy':<16} {'Return':>9} {'Value':>12} {'Trades':>7} {'WinRate':>8} {'MaxDD':>8} {'Sharpe':>7} {'Pos':>6}")
        print(f"  {'-'*73}")
        for ss in st.strategies.values():
            pos = "LONG" if ss.position == 1 else "FLAT"
            val = ss.portfolio_values[-1] if ss.portfolio_values else 0
            print(
                f"  {ss.name:<16} {ss.current_return:>+8.2%} ${val:>10,.2f} "
                f"{ss.num_trades:>7d} {ss.win_rate:>7.1%} {ss.max_drawdown:>+7.2%} "
                f"{ss.sharpe:>7.2f} {pos:>6}"
            )
        print(f"  {'-'*73}")
        if st.prices:
            print(f"  {'Buy & Hold':<16} {st.price_return:>+8.2%}")
        print(f"{'='*80}")
