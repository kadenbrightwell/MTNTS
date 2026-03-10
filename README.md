# Multi-Ticker Neural Trading System

Neural network-based trading simulation for **UVXY**, **SPXU**, **SVIX**, and **SPXL** using per-ticker ensemble models with CUDA acceleration and a 35-ticker feature universe.

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
cd MTNTS
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
python scripts/fetch_data.py --full       # 1. Download all 35 tickers
python scripts/train.py --model transformer --seeds 20 --seq-len 30
                                           # 2. Train transformer ensembles
python scripts/backtest.py --monte-carlo 5000
                                           # 3. Backtest with strategy grid + significance
python scripts/live.py --replay --replay-hours 168 --replay-interval 1m
                                           # 4. Replay last 7 days (1-minute)
```

---

## Recommended Commands

### Quick Training (fast iteration)

Good for rapid experimentation. Trains 10 seeds per ticker with moderate sequence length.

```bash
python scripts/train.py --model transformer --seeds 10 --seq-len 30 --epochs 1000 --patience 50
```

### Strong Training (best quality)

Maximum model quality. More seeds (top half are kept automatically), deeper CV, and longer patience.

```bash
python scripts/train.py --model transformer --seeds 30 --cv-folds 8 --seq-len 30 --epochs 2000 --patience 80
```

### Full Backtest with Significance

Runs all 4 tickers, 7-strategy grid, and 5000 Monte Carlo simulations for each.

```bash
python scripts/backtest.py --monte-carlo 5000 --grid
```

### Quick Single-Ticker Backtest

```bash
python scripts/backtest.py --ticker UVXY --no-grid
```

### Parallel Backtest

Run all 4 tickers concurrently (useful when Monte Carlo is high):

```bash
python scripts/backtest.py --monte-carlo 5000 --workers 4
```

---

## Tuning for Better Performance

### Sequence Length (`--seq-len`)

Controls how many days of history the model sees per prediction. Default is **30**.

- **15-20**: Faster training, captures short-term patterns, works with small datasets
- **30** (default): Good balance of context and data efficiency
- **60**: More context for long-term patterns, needs ~900+ samples

### Number of Seeds (`--seeds`)

Each seed trains an independent model. The ensemble averages predictions across all kept models. After training, the worst 50% of seeds are automatically pruned.

- **10**: Fast iteration (keeps best 5)
- **20** (recommended): Good ensemble diversity (keeps best 10)
- **30-40**: Maximum quality (keeps best 15-20)

### Feature Cap

The system generates **900+ raw features** from 35 tickers, prunes correlated/low-variance features, then selects the **top 80** by target correlation. This is automatic and requires no tuning.

### Epochs and Patience

- `--epochs`: Maximum training epochs (default 1000). Rarely reached due to early stopping.
- `--patience`: How many epochs without improvement before stopping (default 50).
- Increasing patience to 80-100 can help on larger datasets.

### Stochastic Weight Averaging (SWA)

Enabled automatically. The trainer averages the last 5 best checkpoint weights, smoothing the loss surface and improving generalization.

---

## Commands

### Fetch Data

Downloads OHLCV history for all 35 tickers into a local SQLite database.

```bash
python scripts/fetch_data.py              # incremental update
python scripts/fetch_data.py --full       # full re-download from 2022-03-30
```

### Train

Trains per-ticker multi-seed ensembles with walk-forward cross-validation.

```bash
# All 4 tickers, default settings
python scripts/train.py

# Single ticker with transformer
python scripts/train.py --model transformer --ticker UVXY

# All options
python scripts/train.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `lstm` | `lstm` or `transformer` |
| `--seeds` | `10` | Ensemble seeds per ticker (top half kept) |
| `--cv-folds` | `4` | Walk-forward CV folds (0 to skip) |
| `--epochs` | `1000` | Max epochs per seed |
| `--patience` | `50` | Early stopping patience |
| `--seq-len` | `30` | Lookback window |
| `--ticker` | all 4 | Train a single ticker |

### Backtest

Per-ticker strategy grids with Monte Carlo significance testing.

```bash
python scripts/backtest.py                       # all 4 tickers, full grid
python scripts/backtest.py --ticker SPXL         # single ticker
python scripts/backtest.py --monte-carlo 5000    # 5000 MC simulations
python scripts/backtest.py --workers 4           # parallel tickers
python scripts/backtest.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--capital` | `10000` | Total starting capital (split across tickers) |
| `--monte-carlo` | `500` | Monte Carlo simulations (0 to skip) |
| `--grid / --no-grid` | `--grid` | Multi-strategy grid |
| `--ticker` | all 4 | Backtest a single ticker |
| `--workers` | `1` | Parallel workers for multi-ticker backtests |

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

# Instant replay (processes all ticks, renders final dashboard)
python scripts/live.py --replay --replay-hours 168 --replay-interval 1m --replay-speed 0

# 60 days of 5-minute data
python scripts/live.py --replay --replay-hours 1440 --replay-interval 5m --replay-speed 0.1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--replay` | off | Enable replay mode |
| `--replay-hours` | `24` | Hours of history |
| `--replay-interval` | `1m` | Candle size |
| `--replay-speed` | `0.0` | Seconds per tick (0 = instant) |

**Yahoo Finance data limits:**

| Interval | Max history |
|----------|-------------|
| `1m` | 7 days |
| `2m` - `30m` | 60 days |
| `1h` | ~2 years |

---

## Feature Universe (35 Tickers)

### Targets (4)

| Ticker | Description |
|--------|-------------|
| UVXY | 1.5x Long VIX Futures |
| SPXU | 3x Inverse S&P 500 |
| SVIX | -1x Short VIX Futures |
| SPXL | 3x Long S&P 500 |

### Features (31)

| Category | Tickers | Purpose |
|----------|---------|---------|
| **S&P 500 Index** | SPY, QQQ, IWM, DIA | Direct underlier for SPXL/SPXU |
| **Volatility** | ^VIX, VIXY, VIXM, ^VIX9D | Direct underlier for UVXY/SVIX, term structure, near-term sentiment |
| **Fixed Income** | TLT, IEF, SHY, HYG, LQD, ^TNX, ^IRX, ^TYX | Full yield curve (3M/10Y/30Y), credit risk |
| **Commodities** | GLD, USO, SLV | Safe haven, economic activity |
| **Currency** | UUP | Dollar strength |
| **Sectors** | XLF, XLE, XLK, XLU, XLV, XLY, XLP, XLI | Full sector rotation signals, defensive vs cyclical |
| **Global** | EEM, EFA | International risk appetite, EM vs DM |
| **Real Estate** | VNQ | Rate sensitivity, risk-on barometer |
| **Crypto** | BTC-USD | Risk-on barometer |

### Derived Features (~900+ raw, top 80 selected)

**Per-ticker price features** (14 per ticker):
- Log returns, SMA ratios (5/10/21d), rolling volatility, Bollinger width, lagged returns

**Per-ticker volume features** (3 per ticker):
- Volume/SMA ratio (surge detection), volume ROC, OBV slope (accumulation/distribution)

**Per-ticker technical indicators** (8 per ticker):
- RSI(14), MACD histogram, Stochastic %K, ATR%, ADX(14), CCI(20), Williams %R(14), ROC(10)

**Cross-asset signals** (~35 features):
- VIX term structure (VIXY/VIXM), VIX9D/VIX near-term ratio
- VIX regime buckets (low/mid/high/extreme) and rate-of-change
- Full yield curve slopes (10Y-3M, 30Y-10Y)
- Credit spread proxy (HYG/LQD), VIX x credit interaction
- Market breadth (IWM/SPY), risk sentiment (BTC/GLD)
- Sector relative strength (8 sectors vs SPY), sector momentum spread, defensive vs cyclical
- Equity/bond flight-to-safety (TLT/SPY ratio)
- EM and international relative strength (EEM/SPY, EFA/SPY)
- Pair deviation signals (UVXY*SVIX, SPXL*SPXU with z-scores)
- Rolling correlations and beta vs SPY
- Dollar impact (UUP/SPY)

**Calendar features** (5 features):
- Day-of-week (sine/cosine encoded), month (sine/cosine encoded), options expiration week flag

---

## Project Structure

```
MTNTS/
├── config.py                 # All config, paths, 35-ticker universe, device
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── scripts/                  # CLI entry points
│   ├── fetch_data.py         # Download 35-ticker market data
│   ├── train.py              # Per-ticker ensemble training + CV + seed pruning
│   ├── backtest.py           # Per-ticker strategy grid backtest (parallel)
│   └── live.py               # Multi-ticker live simulation & replay
│
├── src/                      # Library code
│   ├── data/
│   │   ├── fetcher.py        # yfinance data acquisition
│   │   ├── storage.py        # SQLite read/write
│   │   └── preprocessor.py   # Feature engineering, scaling, datasets
│   ├── model/
│   │   ├── architecture.py   # LSTM-Attention and Transformer models
│   │   ├── trainer.py        # Training loop, early stopping, SWA
│   │   └── predictor.py      # Per-ticker and multi-ticker inference
│   ├── backtest/
│   │   ├── engine.py         # Single and multi-ticker backtest engines
│   │   └── metrics.py        # Performance metrics + Monte Carlo
│   └── live/
│       ├── runner.py         # Multi-ticker live and replay orchestration
│       └── dashboard.py      # Rich + plotext multi-ticker terminal UI
│
├── data/                     # (created at runtime, gitignored)
│   ├── raw/market.db         # SQLite database (35 tables)
│   └── processed/            # Per-ticker subdirectories
│       ├── uvxy/             # scaler.pkl, feature_cols.pkl
│       ├── spxu/
│       ├── svix/
│       └── spxl/
│
└── models/                   # (created at runtime, gitignored)
    ├── uvxy/                 # best_model_seed*.pt (top K kept)
    ├── spxu/
    ├── svix/
    └── spxl/
```

---

## Architecture

- **Data pipeline:** yfinance (35 tickers) -> SQLite -> feature engineering (900+ raw features: log returns, rolling stats, volume analysis, RSI, MACD, Stochastic, ATR, ADX, CCI, Williams %R, ROC, VIX term structure, full yield curve, credit spreads, sector rotation, pair deviations, cross-asset correlations, calendar effects) -> correlation/variance pruning -> top-80 selection -> RobustScaler -> PyTorch Dataset
- **Models:** Per-ticker Bidirectional LSTM with multi-head self-attention, or Temporal Transformer encoder (2 layers, 128 hidden, 4 heads)
- **Training:** Mixed-precision FP16, AdamW, warmup + cosine annealing, HuberLoss, early stopping, gradient clipping, Stochastic Weight Averaging (SWA)
- **Augmentation:** Gaussian noise, random feature masking (7%), time jitter
- **Ensemble:** Multiple seeds trained independently per ticker, worst 50% pruned by loss, remaining predictions averaged
- **Signal processing:** Per-ticker rolling z-score normalization adapts daily-trained predictions to minute-level timescales
- **Strategies:** 7 preset threshold levels run simultaneously for comparison, each managing independent positions across all 4 tickers
- **Backtesting:** Monte Carlo significance testing, parallel multi-ticker execution, per-ticker and portfolio summary metrics
