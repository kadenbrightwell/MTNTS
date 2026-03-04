"""Central configuration for the ETHU Neural Trading System."""

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
TARGET_TICKER = "ETHU"
FEATURE_TICKERS_ETF = ["GLD", "SLV", "USO", "SPY"]
FEATURE_TICKERS_CRYPTO = ["BTC-USD", "ETH-USD"]
ALL_TICKERS = [TARGET_TICKER] + FEATURE_TICKERS_ETF + FEATURE_TICKERS_CRYPTO

TICKER_DISPLAY = {
    "ETHU": "ETHU", "GLD": "GLD", "SLV": "SLV",
    "USO": "USO", "SPY": "SPY",
    "BTC-USD": "BTC", "ETH-USD": "ETH",
}

DATA_START_DATE = "2024-06-04"

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
            "Install PyTorch with CUDA 12.8+ for RTX 5070ti acceleration.\n"
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
    seq_len: int = 15
    hidden_size: int = 64
    num_layers: int = 1
    num_heads: int = 4
    dim_feedforward: int = 128
    dropout: float = 0.4
    fc_hidden: int = 64


@dataclass
class TrainConfig:
    epochs: int = 500
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    patience: int = 30
    grad_clip: float = 1.0
    warmup_epochs: int = 10
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
    interval_minutes: int = 5
    initial_capital: float = 10_000.0
    intraday_interval: str = "5m"
    long_threshold: float = 0.001
    flat_threshold: float = -0.001
