# Multi-Ticker Neural Trading System

Neural network-based trading simulation for **UVXY**, **SPXU**, **SVIX**, and **SPXL** using per-ticker LSTM-Attention ensembles with CUDA acceleration and a 27-ticker feature universe.

## Instruments

| Ticker | Name | Leverage | Underlier |
|--------|------|----------|-----------|
| **UVXY** | ProShares Ultra VIX Short-Term Futures | 1.5x Long | VIX Futures |
| **SVIX** | VS Trust Short VIX Futures ETF | -1x Short | VIX Futures |
| **SPXL** | Direxion Daily S&P 500 Bull 3X | 3x Long | S&P 500 |
| **SPXU** | ProShares UltraPro Short S&P500 | 3x Inverse | S&P 500 |

These form two inverse pairs: **UVXY/SVIX** (volatility) and **SPXL/SPXU** (equity). Each ticker gets its own independent model ensemble.

---

## Prerequisites

- **Python** 3.11 or newer
- **NVIDIA GPU** with CUDA support (any modern card works; tested on RTX 5070 Ti)
- **NVIDIA drivers** installed ([download](https://www.nvidia.com/drivers))

---

## Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd "UVXY - SPXU -=- SVIX - SPXL"
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 3. Install PyTorch with CUDA

```bash
# CUDA 12.8+ (RTX 40/50-series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# CPU only
pip install torch torchvision torchaudio
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify GPU detection

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None"
```

---

## Quick Start

```bash
python scripts/fetch_data.py --full       # 1. Download all 27 tickers
python scripts/train.py                    # 2. Train ensembles for all 4 targets
python scripts/backtest.py                 # 3. Backtest with strategy grid
python scripts/live.py --replay --replay-hours 168 --replay-interval 1m --replay-speed 0.5
                                           # 4. Replay last 7 days (1-minute)
```

---

## Commands

### Fetch Data

Downloads OHLCV history for all 27 tickers into a local SQLite database.

```bash
python scripts/fetch_data.py              # incremental update
python scripts/fetch_data.py --full       # full re-download from 2022-03-30
```

### Train

Trains per-ticker multi-seed ensembles with walk-forward cross-validation.

```bash
# All 4 tickers, 10-seed ensembles, 4-fold CV
python scripts/train.py

# Single ticker
python scripts/train.py --ticker UVXY

# Transformer model, more seeds
python scripts/train.py --model transformer --seeds 20

# All options
python scripts/train.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `lstm` | `lstm` or `transformer` |
| `--seeds` | `10` | Ensemble seeds per ticker |
| `--cv-folds` | `4` | Walk-forward CV folds (0 to skip) |
| `--epochs` | `500` | Max epochs per seed |
| `--patience` | `30` | Early stopping patience |
| `--seq-len` | `15` | Lookback window |
| `--ticker` | all 4 | Train a single ticker |

### Backtest

Per-ticker strategy grids with Monte Carlo significance testing.

```bash
python scripts/backtest.py                # all 4 tickers, full grid
python scripts/backtest.py --ticker SPXL  # single ticker
python scripts/backtest.py --monte-carlo 2000
python scripts/backtest.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--capital` | `10000` | Total starting capital (split across tickers) |
| `--monte-carlo` | `500` | Monte Carlo simulations (0 to skip) |
| `--grid / --no-grid` | `--grid` | Multi-strategy grid |
| `--ticker` | all 4 | Backtest a single ticker |

### Live Simulation

Real-time terminal dashboard with minute-by-minute trading across all 4 tickers.

```bash
python scripts/live.py --duration 8         # 8 hours, 1-minute intervals
python scripts/live.py --duration 24        # overnight
python scripts/live.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--duration` | `4` | Hours to run |
| `--interval` | `1` | Minutes between data fetches |
| `--capital` | `10000` | Starting capital per strategy |

### Historical Replay

```bash
# Replay last 7 days of 1-minute data
python scripts/live.py --replay --replay-hours 168 --replay-interval 1m --replay-speed 0.5

# Instant replay
python scripts/live.py --replay --replay-hours 168 --replay-interval 1m --replay-speed 0

# 60 days of 5-minute data
python scripts/live.py --replay --replay-hours 1440 --replay-interval 5m --replay-speed 0.1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--replay` | off | Enable replay mode |
| `--replay-hours` | `24` | Hours of history |
| `--replay-interval` | `1m` | Candle size |
| `--replay-speed` | `1.0` | Seconds per tick (0 = instant) |

**Yahoo Finance data limits:**

| Interval | Max history |
|----------|-------------|
| `1m` | 7 days |
| `2m` - `30m` | 60 days |
| `1h` | ~2 years |

---

## Feature Universe (27 Tickers)

### Targets (4)

| Ticker | Description |
|--------|-------------|
| UVXY | 1.5x Long VIX Futures |
| SPXU | 3x Inverse S&P 500 |
| SVIX | -1x Short VIX Futures |
| SPXL | 3x Long S&P 500 |

### Features (23)

| Category | Tickers | Purpose |
|----------|---------|---------|
| **S&P 500 Index** | SPY, QQQ, IWM, DIA | Direct underlier for SPXL/SPXU |
| **Volatility** | ^VIX, VIXY, VIXM | Direct underlier for UVXY/SVIX, term structure |
| **Fixed Income** | TLT, IEF, SHY, HYG, LQD, ^TNX | Rate expectations, credit risk |
| **Commodities** | GLD, USO, SLV | Safe haven, economic activity |
| **Currency** | UUP | Dollar strength |
| **Sectors** | XLF, XLE, XLK, XLU | Sector rotation signals |
| **Global** | EEM, EFA | International risk appetite |
| **Crypto** | BTC-USD | Risk-on barometer |

### Derived Features

- VIX term structure (VIXY/VIXM contango/backwardation)
- Yield curve slope (TLT/SHY)
- Credit spread proxy (HYG/LQD)
- Market breadth (IWM/SPY)
- Risk sentiment (BTC/GLD)
- Sector relative strength (XLF/SPY, XLE/SPY, XLK/SPY, XLU/SPY)
- Pair deviation signals (UVXY*SVIX, SPXL*SPXU)
- VIX regime buckets and rate of change
- Rolling cross-asset correlations and betas

---

## Project Structure

```
UVXY - SPXU -=- SVIX - SPXL/
├── config.py                 # All config, paths, ticker universe, device
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── scripts/                  # CLI entry points
│   ├── fetch_data.py         # Download 27-ticker market data
│   ├── train.py              # Per-ticker ensemble training + CV
│   ├── backtest.py           # Per-ticker strategy grid backtest
│   └── live.py               # Multi-ticker live simulation & replay
│
├── src/                      # Library code
│   ├── data/
│   │   ├── fetcher.py        # yfinance data acquisition
│   │   ├── storage.py        # SQLite read/write
│   │   └── preprocessor.py   # Feature engineering, scaling, datasets
│   ├── model/
│   │   ├── architecture.py   # LSTM-Attention and Transformer models
│   │   ├── trainer.py        # Training loop, early stopping
│   │   └── predictor.py      # Per-ticker and multi-ticker inference
│   ├── backtest/
│   │   ├── engine.py         # Single and multi-ticker backtest engines
│   │   └── metrics.py        # Performance metrics + Monte Carlo
│   └── live/
│       ├── runner.py         # Multi-ticker live and replay orchestration
│       └── dashboard.py      # Rich + plotext multi-ticker terminal UI
│
├── data/                     # (created at runtime, gitignored)
│   ├── raw/market.db         # SQLite database (27 tables)
│   └── processed/            # Per-ticker subdirectories
│       ├── uvxy/             # scaler.pkl, feature_cols.pkl
│       ├── spxu/
│       ├── svix/
│       └── spxl/
│
└── models/                   # (created at runtime, gitignored)
    ├── uvxy/                 # best_model_seed*.pt
    ├── spxu/
    ├── svix/
    └── spxl/
```

---

## Architecture

- **Data pipeline:** yfinance (27 tickers) -> SQLite -> feature engineering (log returns, rolling stats, RSI, MACD, Stochastic, ATR, VIX term structure, yield curve, credit spreads, sector rotation, pair deviations, cross-asset correlations) -> RobustScaler -> PyTorch Dataset
- **Models:** Per-ticker Bidirectional LSTM with multi-head self-attention (default), or Temporal Transformer encoder
- **Training:** Mixed-precision FP16, AdamW, warmup + cosine annealing, HuberLoss, early stopping, gradient clipping
- **Ensemble:** Multiple seeds trained independently per ticker, predictions averaged
- **Signal processing:** Per-ticker rolling z-score normalization adapts daily-trained predictions to minute-level timescales
- **Strategies:** 5 preset threshold levels run simultaneously for comparison, each managing independent positions across all 4 tickers
