# QuantumEdge — Algorithmic Trading Framework

> A data-driven 5-minute cryptocurrency trading system for Binance markets.
> Python-first prototyping with numba-accelerated backtesting.
> 96 tests across 15 modules, all passing.

---

## Overview

QuantumEdge is a quantitative trading framework targeting Binance cryptocurrency markets on the 5-minute timeframe. It leverages over 10 years of historical OHLCV data to discover and validate repeatable statistical edges.

**Development timeline:** May 2026 (4 sessions, ~3 hours of active development)

### Key Findings

| Strategy | Sharpe | PF | Max DD | Period |
|---|---|---|---|---|
| RSI Mean Reversion (25, 30, 88) | **+0.31** | 1.12 | -6.45% | 2024-2025 OOS |
| RSI Mean Reversion (14, 30, 70) | -0.21 | 0.75 | -56.93% | 2017-2025 |
| MA Crossover | -9.27 | 0.65 | -100% | 2017-2025 |
| Donchian Breakout | -7.48 | 0.63 | -100% | 2017-2025 |

**Best strategy:** RSI Mean Reversion with long period (25) and wide thresholds (oversold 30, overbought 88). Positive Sharpe on 2024-2025 out-of-sample data with controlled -6.45% drawdown.

---

## Architecture

```
├── AGENTS.md                    Agent instructions
├── todo.md                      Task tracker
├── docs/                        Whitepaper, strategy guidelines, backtesting standards
├── data/
│   ├── btc-yearly/              Raw JSON (9 files, 126 MB, 2017-2025)
│   └── processed/               Parquet (1 file, 31 MB, 876,672 rows)
├── research/
│   ├── data/loader.py           Data loader, validator, Parquet I/O
│   ├── features/
│   │   ├── indicators.py        SMA, EMA, RSI, ATR, Bollinger Bands
│   │   └── ml_features.py       35+ feature ML matrix generator
│   ├── backtest/
│   │   ├── engine.py            Vectorized backtest engine (numba accelerated)
│   │   ├── metrics.py           Sharpe, Sortino, Calmar, PF, DD, etc.
│   │   └── _numba_ops.py        Numba trade extraction
│   ├── strategies/
│   │   ├── base.py              Abstract strategy interface
│   │   ├── ma_crossover.py      Moving Average Crossover
│   │   ├── breakout.py          Donchian Channel Breakout (numba)
│   │   ├── mean_reversion.py    RSI / Bollinger Mean Reversion (numba)
│   │   ├── volatility_breakout.py  ATR Volatility Breakout (numba)
│   │   ├── regime.py            SMA / ADX Regime Detection + filter
│   │   ├── ensemble.py          Rolling Sharpe-weighted signal combiner
│   │   └── _numba_ops.py        4 numba-accelerated state machines
│   ├── ml/
│   │   ├── trainer.py           LightGBM pipeline (32 threads)
│   │   ├── importance.py        Feature importance reports
│   │   └── models/              Trained model files
│   ├── optimization/
│   │   ├── search.py            Random search + walk-forward parameter optimization
│   │   └── robustness.py        Monte Carlo, regime breakdown, parameter stability
│   └── execution/
│       └── paper.py             Paper trading pipeline (cron-ready)
├── tests/                       96 tests across 15 test files
└── results/
    ├── reports/                 Optimization reports, robustness reports, feature importance
    └── paper/                   Paper trading logs
```

---

## Setup

```bash
# Create venv (git-bash on Windows)
python -m venv .venv
source .venv/Scripts/activate

# Install dependencies
pip install pandas numpy python-binance vectorbt pytest pytest-cov \
            jupyter matplotlib seaborn pyarrow lightgbm

# Load & process data
python -c "
from research.data.loader import load_all, save_parquet
df = load_all()
save_parquet(df)
print(f'Loaded {len(df):,} rows')
"

# Run tests
python -m pytest -v
```

---

## Performance

| Benchmark | Time | Dataset |
|---|---|---|
| MA Crossover signal gen | **0.03s** | 876K rows |
| Donchian Breakout signal gen (numba) | **0.45s** | 876K rows |
| Full backtest (signal + metrics) | **~0.3-1.5s** | 876K rows |
| Parameter sweep (single iteration) | **26ms** | 100K rows |
| ML training (50 rounds, 220K train) | **~15s** | 32 threads |
| Full test suite (96 tests) | **~37s** | All modules |

**Hardware:** 32-thread CPU, 96 GB DDR5 RAM, 7000 MB/s NVMe SSD

---

## Usage

### Run a Strategy
```python
from research.data.loader import load_parquet
from research.strategies.mean_reversion import MeanReversion
from research.backtest.engine import VectorizedBacktest
from research.backtest.metrics import compute_metrics, format_metrics_report

data = load_parquet()
strat = MeanReversion({"mode": "rsi", "rsi_period": 25, "rsi_oversold": 30, "rsi_overbought": 88})
signals = strat.generate_signals(data)
result = VectorizedBacktest(data).run(signals)
metrics = compute_metrics(result.equity_curve, result.trades)
print(format_metrics_report(metrics))
```

### Parameter Optimization
```python
from research.optimization.search import run_mean_reversion_search, generate_optimization_report
from pathlib import Path
results = run_mean_reversion_search(data, n=50)
generate_optimization_report(results, "Mean Reversion", Path("results/reports"))
```

### Paper Trading
```bash
python -c "from research.execution.paper import paper_trade_job; print(paper_trade_job())"
```

### Ensemble
```python
from research.strategies.ensemble import ensemble_signals
combined = ensemble_signals(signal_dict, data["close"], method="sharpe_weight")
```

---

## Test Suite

```
96 passed in 37s
```

| Module | Tests |
|---|---|
| Data loader | 14 |
| Indicators | 6 |
| Base Strategy | 7 |
| MA Crossover | 5 |
| Donchian Breakout | 7 |
| Mean Reversion | 6 |
| Volatility Breakout | 5 |
| Regime Detection | 7 |
| Ensemble | 6 |
| ML Features | 10 |
| ML Trainer | 3 |
| Backtest Engine | 12 |
| Optimization | 3 |
| Robustness | 3 |
| Paper Trading | 1 |
| Backtest Metrics | (in engine tests) |

---

## Project Status

| Phase | Status | Deliverables |
|---|---|---|
| 1: Foundation | ✅ | Data loader, EDA, backtest engine, metrics |
| 2: Classical | ✅ | 5 strategies + regime + ensemble (numba) |
| 3: Advanced | ✅ | ML pipeline + feature importance |
| 4: Validation | ✅ | Optimization, robustness, paper trading |
| Hardware Opt | ✅ | Numba acceleration, 32-thread ML |

---

## License

MIT
