# ETHU Neural Trading System

Neural network-based trading simulation for **ETHU** (2x Ether ETF) using an LSTM-Attention ensemble with CUDA acceleration.

---

## Prerequisites

- **Python** 3.11 or newer
- **NVIDIA GPU** with CUDA support (any modern card works; tested on RTX 5070 Ti)
- **NVIDIA drivers** installed ([download](https://www.nvidia.com/drivers))

---

## Setup (New Machine)

### 1. Clone the repo

```bash
git clone <your-repo-url> ETHU
cd ETHU
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 3. Install PyTorch with CUDA

Pick the command that matches your GPU's CUDA version. Check yours with `nvidia-smi`.

**CUDA 12.8+ (RTX 40-series, 50-series, Blackwell):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**CUDA 12.4 (RTX 30-series, 40-series):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**CUDA 12.1 (older cards):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Nightly (if your GPU arch isn't in stable yet):**
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**CPU only (no GPU):**
```bash
pip install torch torchvision torchaudio
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify GPU detection

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None"
```

You should see your GPU name. If not, your PyTorch CUDA version doesn't match your driver.

---

## Quick Start

Run these four commands in order on a fresh machine:

```bash
python scripts/fetch_data.py            # 1. Download market data
python scripts/train.py                  # 2. Train ensemble (10 models)
python scripts/backtest.py               # 3. Backtest with strategy grid
python scripts/live.py --replay --replay-hours 168 --replay-interval 1m --replay-speed 0.5
                                         # 4. Replay last 7 days
```

---

## Commands Reference

### Fetch Data

Downloads OHLCV history for ETHU, GLD, SLV, USO, SPY, BTC-USD, ETH-USD into a local SQLite database.

```bash
python scripts/fetch_data.py              # incremental update (fast)
python scripts/fetch_data.py --full       # full re-download from 2024-06-04
```

Run this before training or backtesting, and periodically to keep data current.

### Train

Trains a multi-seed ensemble with walk-forward cross-validation. All artifacts (model checkpoints, scaler, feature columns) are saved to `models/` and `data/processed/`.

```bash
# Default: 10-seed ensemble, 4-fold CV, LSTM, 500 epochs max
python scripts/train.py

# Scale up for a more powerful machine
python scripts/train.py --seeds 20 --cv-folds 6 --epochs 1000 --patience 50

# Transformer model instead of LSTM
python scripts/train.py --model transformer --seeds 15

# All options
python scripts/train.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `lstm` | `lstm` or `transformer` |
| `--seeds` | `10` | Number of ensemble models to train |
| `--cv-folds` | `4` | Walk-forward CV folds (0 to skip) |
| `--epochs` | `500` | Max epochs per seed (early stopping) |
| `--patience` | `30` | Early stopping patience |
| `--seq-len` | `15` | Lookback window (days) |
| `--batch-size` | `32` | Training batch size |
| `--lr` | `0.0005` | Learning rate |

**Scaling guidance:** More seeds = more robust ensemble. More CV folds = better validation. More epochs + patience = let each seed converge fully. On a powerful GPU, `--seeds 20 --epochs 1000 --patience 60` is a good starting point.

### Backtest

Runs all trained models against historical data with a grid of 7 strategy configurations and Monte Carlo significance testing.

```bash
# Default: full grid + 500 Monte Carlo simulations
python scripts/backtest.py

# More Monte Carlo iterations for tighter p-values
python scripts/backtest.py --monte-carlo 2000

# Single strategy only
python scripts/backtest.py --no-grid --capital 50000

# Backtest from a specific date
python scripts/backtest.py --start 2025-01-01

# All options
python scripts/backtest.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--capital` | `10000` | Starting capital |
| `--monte-carlo` | `500` | Monte Carlo simulations (0 to skip) |
| `--grid / --no-grid` | `--grid` | Run multi-strategy grid |
| `--start` | all data | Backtest start date |
| `--model-path` | auto-detect | Path to model checkpoint |

### Live Simulation

Real-time terminal dashboard that fetches live market data and simulates trades.

```bash
# Run for 8 hours, checking every 5 minutes
python scripts/live.py --duration 8 --interval 5

# Run overnight (24 hours)
python scripts/live.py --duration 24

# Run for a week
python scripts/live.py --duration 168

# Higher capital
python scripts/live.py --duration 12 --capital 50000

# All options
python scripts/live.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--duration` | `4` | Hours to run |
| `--interval` | `5` | Minutes between data fetches |
| `--capital` | `10000` | Starting capital per strategy |

There is no maximum duration. Leave it running for days or weeks.

### Historical Replay

Replays past market data through the model at accelerated speed. Same engine as live mode but uses pre-fetched historical candles.

```bash
# Replay last 7 days of 1-minute data at 0.5s per tick
python scripts/live.py --replay --replay-hours 168 --replay-interval 1m --replay-speed 0.5

# Instant replay (as fast as possible)
python scripts/live.py --replay --replay-hours 168 --replay-interval 1m --replay-speed 0

# Replay 60 days of 5-minute data
python scripts/live.py --replay --replay-hours 1440 --replay-interval 5m --replay-speed 0.1

# Replay last 24 hours at real speed
python scripts/live.py --replay --replay-hours 24 --replay-interval 1m --replay-speed 60

# All options
python scripts/live.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--replay` | off | Enable replay mode |
| `--replay-hours` | `24` | Hours of history to replay |
| `--replay-interval` | `5m` | Candle size: `1m`, `2m`, `5m`, `15m`, `30m`, `1h` |
| `--replay-speed` | `1.0` | Seconds per tick (0 = instant) |

**Data limits (Yahoo Finance):**

| Interval | Max history |
|----------|-------------|
| `1m` | 7 days (168 hours) |
| `2m` - `30m` | 60 days (1440 hours) |
| `1h` | ~2 years |

---

## Project Structure

```
ETHU/
├── config.py                 # All hyperparameters, paths, device detection
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── scripts/                  # CLI entry points
│   ├── fetch_data.py         # Download market data
│   ├── train.py              # Train ensemble + cross-validation
│   ├── backtest.py           # Strategy grid backtest
│   └── live.py               # Live simulation & replay
│
├── src/                      # Library code
│   ├── data/
│   │   ├── fetcher.py        # yfinance data acquisition
│   │   ├── storage.py        # SQLite read/write
│   │   └── preprocessor.py   # Feature engineering, scaling, datasets
│   ├── model/
│   │   ├── architecture.py   # LSTM-Attention and Transformer models
│   │   ├── trainer.py        # Training loop, early stopping, schedulers
│   │   └── predictor.py      # Ensemble inference wrapper
│   ├── backtest/
│   │   ├── engine.py         # Trade simulation engine
│   │   └── metrics.py        # Performance metrics + Monte Carlo
│   └── live/
│       ├── runner.py         # Live and replay orchestration
│       └── dashboard.py      # Rich + plotext terminal UI
│
├── data/                     # (created at runtime, gitignored)
│   ├── raw/market.db         # SQLite database
│   └── processed/            # scaler.pkl, feature_cols.pkl
│
└── models/                   # (created at runtime, gitignored)
    └── best_model_seed*.pt   # Trained checkpoints
```

---

## Architecture

- **Data pipeline:** yfinance → SQLite → feature engineering (log returns, rolling stats, RSI, MACD, Stochastic, ATR, cross-asset ratios, lagged features) → RobustScaler → PyTorch Dataset
- **Model:** Bidirectional LSTM with multi-head self-attention (default), or Temporal Transformer encoder. ~129K parameters.
- **Training:** Mixed-precision FP16, AdamW optimizer, warmup + cosine annealing LR schedule, HuberLoss, early stopping, gradient clipping
- **Ensemble:** Multiple seeds trained independently, predictions averaged for variance reduction
- **Signal processing:** Rolling z-score normalization adapts daily-trained predictions to intraday timescales
- **Strategies:** 5 preset thresholds (Conservative → Ultra-Aggressive) run simultaneously for comparison

## Feature Tickers

| Ticker | Description |
|--------|-------------|
| ETHU | 2x Ether ETF (target) |
| GLD | Gold ETF |
| SLV | Silver ETF |
| USO | Oil ETF |
| SPY | S&P 500 ETF |
| BTC-USD | Bitcoin |
| ETH-USD | Ethereum |
