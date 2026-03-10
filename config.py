"""Central configuration for the Multi-Ticker Neural Trading System.

Trades UVXY, SPXU, SVIX, and SPXL using a 35-ticker feature universe
spanning volatility, equities, rates, credit, commodities, currency,
sectors, global markets, and crypto.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

import torch


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
DB_PATH = RAW_DIR / "market.db"

for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Ticker universe
# ---------------------------------------------------------------------------
TARGET_TICKERS: List[str] = ["UVXY", "SPXU", "SVIX", "SPXL"]

FEATURE_TICKERS_INDEX = ["SPY", "QQQ", "IWM", "DIA"]
FEATURE_TICKERS_VOL = ["^VIX", "VIXY", "VIXM", "^VIX9D"]
FEATURE_TICKERS_BONDS = ["TLT", "IEF", "SHY", "HYG", "LQD", "^TNX", "^IRX", "^TYX"]
FEATURE_TICKERS_COMMODITIES = ["GLD", "USO", "SLV"]
FEATURE_TICKERS_CURRENCY = ["UUP"]
FEATURE_TICKERS_SECTOR = ["XLF", "XLE", "XLK", "XLU", "XLV", "XLY", "XLP", "XLI"]
FEATURE_TICKERS_GLOBAL = ["EEM", "EFA"]
FEATURE_TICKERS_REALESTATE = ["VNQ"]
FEATURE_TICKERS_CRYPTO = ["BTC-USD"]

FEATURE_TICKERS: List[str] = (
    FEATURE_TICKERS_INDEX
    + FEATURE_TICKERS_VOL
    + FEATURE_TICKERS_BONDS
    + FEATURE_TICKERS_COMMODITIES
    + FEATURE_TICKERS_CURRENCY
    + FEATURE_TICKERS_SECTOR
    + FEATURE_TICKERS_GLOBAL
    + FEATURE_TICKERS_REALESTATE
    + FEATURE_TICKERS_CRYPTO
)

ALL_TICKERS: List[str] = TARGET_TICKERS + FEATURE_TICKERS

TICKER_DISPLAY: Dict[str, str] = {
    "UVXY": "UVXY", "SPXU": "SPXU", "SVIX": "SVIX", "SPXL": "SPXL",
    "SPY": "SPY", "QQQ": "QQQ", "IWM": "IWM", "DIA": "DIA",
    "^VIX": "VIX", "VIXY": "VIXY", "VIXM": "VIXM", "^VIX9D": "VIX9D",
    "TLT": "TLT", "IEF": "IEF", "SHY": "SHY",
    "HYG": "HYG", "LQD": "LQD", "^TNX": "TNX", "^IRX": "IRX", "^TYX": "TYX",
    "GLD": "GLD", "USO": "USO", "SLV": "SLV",
    "UUP": "UUP",
    "XLF": "XLF", "XLE": "XLE", "XLK": "XLK", "XLU": "XLU",
    "XLV": "XLV", "XLY": "XLY", "XLP": "XLP", "XLI": "XLI",
    "EEM": "EEM", "EFA": "EFA",
    "VNQ": "VNQ",
    "BTC-USD": "BTC",
}

TICKER_PAIRS: Dict[str, str] = {
    "UVXY": "SVIX",
    "SVIX": "UVXY",
    "SPXL": "SPXU",
    "SPXU": "SPXL",
}

DATA_START_DATE = "2022-03-30"


# ---------------------------------------------------------------------------
# Per-ticker path helpers
# ---------------------------------------------------------------------------
def ticker_model_dir(ticker: str) -> Path:
    """Return the model directory for a specific target ticker."""
    d = MODELS_DIR / ticker.lower()
    d.mkdir(parents=True, exist_ok=True)
    return d


def ticker_processed_dir(ticker: str) -> Path:
    """Return the processed-data directory for a specific target ticker."""
    d = PROCESSED_DIR / ticker.lower()
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Device resolver
# ---------------------------------------------------------------------------
def resolve_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        print("[CONFIG] Forced CPU mode.")
        return torch.device("cpu")

    if not torch.cuda.is_available():
        print(
            "[CONFIG] WARNING: CUDA is not available. "
            "Install PyTorch with CUDA 12.8+ for GPU acceleration.\n"
            "  pip install torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu128",
            file=sys.stderr,
        )
        return torch.device("cpu")

    gpu_name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[CONFIG] CUDA device: {gpu_name} ({mem_gb:.1f} GB)")
    return torch.device("cuda")


DEVICE = resolve_device(force_cpu=("--cpu" in sys.argv))


# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    model_type: str = "lstm"          # "lstm" or "transformer"
    seq_len: int = 30
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.3
    fc_hidden: int = 128


@dataclass
class TrainConfig:
    epochs: int = 1000
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    patience: int = 50
    grad_clip: float = 1.0
    warmup_epochs: int = 15
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    use_amp: bool = True


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    slippage_bps: float = 5.0
    commission_bps: float = 5.0
    long_threshold: float = 0.001
    flat_threshold: float = -0.001


@dataclass
class LiveConfig:
    duration_hours: float = 4.0
    interval_minutes: int = 1
    initial_capital: float = 10_000.0
    intraday_interval: str = "1m"
    long_threshold: float = 0.001
    flat_threshold: float = -0.001
