"""Feature engineering, normalization, and PyTorch Dataset for windowed sequences."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import pandas as pd
import ta
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler

from config import (
    TARGET_TICKER, ALL_TICKERS, PROCESSED_DIR,
    FEATURE_TICKERS_CRYPTO, ModelConfig,
)


# ---------------------------------------------------------------------------
# Feature engineering (dict-based to avoid DataFrame fragmentation)
# ---------------------------------------------------------------------------

def _ticker_features(ticker: str, close: pd.Series) -> Dict[str, pd.Series]:
    """Derive price-relative features from a single ticker's close series."""
    pfx = ticker.replace("-", "").lower()
    d: Dict[str, pd.Series] = {}

    ret = np.log(close / close.shift(1))
    d[f"{pfx}_logret"] = ret

    for w in (5, 10, 21):
        sma = close.rolling(w).mean()
        d[f"{pfx}_sma_ratio_{w}"] = close / sma - 1
        d[f"{pfx}_rvol_{w}"] = close.rolling(w).std() / close

    sma21 = close.rolling(21).mean()
    std21 = close.rolling(21).std()
    d[f"{pfx}_bbwidth"] = (2 * std21) / sma21.where(sma21 != 0, np.nan)

    for lag in range(1, 4):
        d[f"{pfx}_ret_lag{lag}"] = ret.shift(lag)

    return d


def _technical_indicators(ohlcv: pd.DataFrame, ticker: str) -> Dict[str, pd.Series]:
    """RSI, MACD, Stochastic, ATR, OBV for a ticker. Returns a dict."""
    pfx = ticker.replace("-", "").lower()
    d: Dict[str, pd.Series] = {}

    if isinstance(ohlcv.columns, pd.MultiIndex):
        high = ohlcv[(ticker, "high")]
        low = ohlcv[(ticker, "low")]
        close = ohlcv[(ticker, "close")]
        volume = ohlcv[(ticker, "volume")]
    else:
        high, low, close, volume = ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]

    d[f"{pfx}_rsi14"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd_ind = ta.trend.MACD(close)
    d[f"{pfx}_macd_hist"] = macd_ind.macd_diff()

    stoch = ta.momentum.StochasticOscillator(high, low, close)
    d[f"{pfx}_stoch_k"] = stoch.stoch()

    atr = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
    d[f"{pfx}_atr_pct"] = atr / close

    return d


def _cross_asset_features(merged: pd.DataFrame) -> Dict[str, pd.Series]:
    """Pairwise rolling correlations, ETH/BTC ratio, SPY-ETHU beta."""
    d: Dict[str, pd.Series] = {}
    closes, rets = {}, {}

    for tkr in ALL_TICKERS:
        if (tkr, "close") in merged.columns:
            c = merged[(tkr, "close")]
            closes[tkr] = c
            rets[tkr] = np.log(c / c.shift(1))

    if "ETH-USD" in closes and "BTC-USD" in closes:
        ratio = closes["ETH-USD"] / closes["BTC-USD"].where(closes["BTC-USD"] != 0, np.nan)
        d["eth_btc_ratio_chg"] = ratio.pct_change()

    target_ret = rets.get(TARGET_TICKER)
    if target_ret is not None:
        for tkr, r in rets.items():
            if tkr == TARGET_TICKER:
                continue
            lbl = tkr.replace("-", "").lower()
            d[f"corr_{lbl}_21d"] = target_ret.rolling(21).corr(r)

    if TARGET_TICKER in rets and "SPY" in rets:
        spy_var = rets["SPY"].rolling(21).var()
        cov = rets[TARGET_TICKER].rolling(21).cov(rets["SPY"])
        d["ethu_spy_beta"] = cov / spy_var.where(spy_var != 0, np.nan)

    return d


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def _prune_correlated(df: pd.DataFrame, threshold: float = 0.85, quiet: bool = False) -> List[str]:
    """Return column names to KEEP after dropping highly correlated features."""
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        if any(upper[col] > threshold):
            to_drop.add(col)
    kept = [c for c in df.columns if c not in to_drop]
    if to_drop and not quiet:
        print(f"  [PREPROCESS] Pruned {len(to_drop)} correlated features (r>{threshold})")
    return kept


def _prune_low_variance(df: pd.DataFrame, threshold: float = 1e-8, quiet: bool = False) -> List[str]:
    """Remove near-constant features."""
    variances = df.var()
    kept = variances[variances > threshold].index.tolist()
    dropped = len(df.columns) - len(kept)
    if dropped and not quiet:
        print(f"  [PREPROCESS] Pruned {dropped} near-zero-variance features")
    return kept


# ---------------------------------------------------------------------------
# Build features
# ---------------------------------------------------------------------------

def build_features(merged: pd.DataFrame, quiet: bool = False) -> Tuple[pd.DataFrame, str]:
    """Build the full feature matrix from merged OHLCV data.

    Returns (features_df, target_col_name).
    """
    ethu_close_raw = merged[(TARGET_TICKER, "close")]
    ethu_mask = ethu_close_raw.notna()
    merged = merged.loc[ethu_mask].copy()
    merged = merged.ffill().bfill()

    parts: Dict[str, pd.Series] = {}

    for tkr in ALL_TICKERS:
        if (tkr, "close") in merged.columns:
            parts.update(_ticker_features(tkr, merged[(tkr, "close")]))

    parts.update(_technical_indicators(merged, TARGET_TICKER))
    parts.update(_cross_asset_features(merged))

    target_col = "target_logret"
    ethu_close = merged[(TARGET_TICKER, "close")]
    parts[target_col] = np.log(ethu_close / ethu_close.shift(1)).shift(-1)

    feat = pd.concat(parts, axis=1)
    feat.index = merged.index

    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat.dropna(inplace=True)

    feature_cols = [c for c in feat.columns if c != target_col]
    kept = _prune_low_variance(feat[feature_cols], quiet=quiet)
    kept = _prune_correlated(feat[kept], quiet=quiet)
    feat = feat[kept + [target_col]]

    if not quiet:
        print(f"  [PREPROCESS] {len(feat)} samples, {len(feat.columns) - 1} features after cleanup")
    return feat, target_col


def _select_top_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    max_features: int = 40,
    quiet: bool = False,
) -> List[int]:
    """Rank features by absolute correlation with target on training data only."""
    if len(feature_names) <= max_features:
        return list(range(len(feature_names)))
    correlations = []
    for i in range(X_train.shape[1]):
        col = X_train[:, i]
        valid = ~(np.isnan(col) | np.isnan(y_train))
        if valid.sum() < 10:
            correlations.append(0.0)
            continue
        if np.std(col[valid]) < 1e-10 or np.std(y_train[valid]) < 1e-10:
            correlations.append(0.0)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r = np.corrcoef(col[valid], y_train[valid])[0, 1]
        correlations.append(abs(r) if not np.isnan(r) else 0.0)
    ranked = np.argsort(correlations)[::-1][:max_features]
    kept = sorted(ranked.tolist())
    if not quiet:
        print(f"  [PREPROCESS] Selected top {max_features} features by target correlation")
    return kept


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def fit_scaler(X_train: np.ndarray) -> RobustScaler:
    scaler = RobustScaler()
    scaler.fit(X_train)
    return scaler


def save_scaler(scaler: RobustScaler, path: Optional[Path] = None) -> Path:
    path = path or (PROCESSED_DIR / "scaler.pkl")
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    return path


def load_scaler(path: Optional[Path] = None) -> RobustScaler:
    path = path or (PROCESSED_DIR / "scaler.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_feature_cols(cols: List[str], path: Optional[Path] = None) -> Path:
    path = path or (PROCESSED_DIR / "feature_cols.pkl")
    with open(path, "wb") as f:
        pickle.dump(cols, f)
    return path


def load_feature_cols(path: Optional[Path] = None) -> List[str]:
    path = path or (PROCESSED_DIR / "feature_cols.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    """Sliding-window dataset producing (X, y) tensors.

    X shape: (seq_len, n_features)
    y shape: scalar (target return for the last timestep in the window)

    When noise_std > 0 and training=True, Gaussian noise is added to features
    each time a sample is drawn, acting as data augmentation.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_len: int = 15,
        noise_std: float = 0.0,
        training: bool = False,
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seq_len = seq_len
        self.noise_std = noise_std
        self.training = training

    def __len__(self) -> int:
        return max(len(self.features) - self.seq_len, 0)

    def __getitem__(self, idx: int):
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len - 1]
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x, y


# ---------------------------------------------------------------------------
# End-to-end helpers
# ---------------------------------------------------------------------------

def prepare_datasets(
    merged: pd.DataFrame,
    cfg: ModelConfig | None = None,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    scaler: Optional[RobustScaler] = None,
) -> Tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset, RobustScaler, list]:
    """Full pipeline: features -> split -> scale -> datasets.

    Returns (train_ds, val_ds, test_ds, scaler, feature_names).
    """
    cfg = cfg or ModelConfig()
    feat_df, target_col = build_features(merged)

    feature_cols = [c for c in feat_df.columns if c != target_col]
    X_all = feat_df[feature_cols].values.astype(np.float32)
    y_all = feat_df[target_col].values.astype(np.float32)

    n = len(X_all)
    if n == 0:
        raise ValueError(
            "Feature matrix is empty after preprocessing. "
            "Ensure data has been fetched: python scripts/fetch_data.py --full"
        )

    min_part = cfg.seq_len + 1
    if n < min_part * 3:
        train_ratio, val_ratio = 0.70, 0.15
        print(f"  [PREPROCESS] Small dataset ({n} rows), using 70/15/15 split")

    n_train = max(int(n * train_ratio), min_part)
    n_val = max(int(n * val_ratio), min_part)
    n_test = n - n_train - n_val

    if n_test < min_part:
        n_val = max((n - n_train) // 2, 1)
        n_test = n - n_train - n_val

    print(f"  [PREPROCESS] Split: train={n_train}, val={n_val}, test={n_test}  (seq_len={cfg.seq_len})")

    X_train_raw = X_all[:n_train]
    y_train = y_all[:n_train]
    X_val_raw = X_all[n_train : n_train + n_val]
    y_val = y_all[n_train : n_train + n_val]
    X_test_raw = X_all[n_train + n_val :]
    y_test = y_all[n_train + n_val :]

    max_feats = 40
    if X_train_raw.shape[1] > max_feats:
        keep_idx = _select_top_features(X_train_raw, y_train, feature_cols, max_feats)
        X_train_raw = X_train_raw[:, keep_idx]
        X_val_raw = X_val_raw[:, keep_idx]
        X_test_raw = X_test_raw[:, keep_idx]
        feature_cols = [feature_cols[i] for i in keep_idx]

    if scaler is None:
        scaler = fit_scaler(X_train_raw)
        save_scaler(scaler)
        save_feature_cols(feature_cols)

    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    seq = cfg.seq_len
    train_ds = TimeSeriesDataset(X_train, y_train, seq, noise_std=0.02, training=True)
    val_ds = TimeSeriesDataset(X_val, y_val, seq)
    test_ds = TimeSeriesDataset(X_test, y_test, seq)

    print(f"  [PREPROCESS] Dataset windows: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_ds, val_ds, test_ds, scaler, feature_cols
