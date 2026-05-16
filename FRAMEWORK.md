# QuantumEdge — Framework Architecture

> A quantitative trading framework for Binance 5-minute cryptocurrency data.
> Python-first prototyping with numba-accelerated backtesting and Monte Carlo optimization.

---

## 1. System Overview

QuantumEdge is structured as a layered research framework. Each layer builds on the one below:

```
┌──────────────────────────────┐
│       Execution Layer       │  Paper trading, live execution
├──────────────────────────────┤
│    Optimization Layer       │  Monte Carlo, walk-forward, robustness
├──────────────────────────────┤
│    Strategy Layer           │  5 strategies + regime + ensemble, numba
├──────────────────────────────┤
│    Backtest Engine          │  Vectorized PnL, 15+ metrics, numba
├──────────────────────────────┤
│    Feature Layer            │  Indicators, ML feature matrix
├──────────────────────────────┤
│    Data Layer               │  Loader, cleaner, fetcher, proxy rotation
└──────────────────────────────┘
```

**Dataset:** 876,672 rows of BTCUSDT 5-minute OHLCV (2017-2025), 31 MB Parquet.
**Tests:** 106 across 16 files, all passing.
**Hardware target:** 32-thread CPU, 96 GB DDR5, NVMe SSD.

---

## 2. Data Layer (`research/data/`)

### 2.1 Data Flow

```
Binance API / JSON files
        │
        ▼
  loader.py
  ├── load_all()       → Reads yearly JSONs, concatenates, validates
  ├── save_parquet()   → Saves as compressed Parquet
  └── load_parquet()   → Loads processed Parquet (primary entry point)
        │
        ▼
  Data cleaning pipeline:
  ├── Timestamp alignment (5-min grid)
  ├── Gap detection & forward-fill (max 2 candles)
  ├── Outlier detection (>50 sigma → NaN)
  └── Duplicate removal
```

### 2.2 Binance API Fetcher

`fetch_binance_klines()` pulls historical klines with:
- Automatic pagination (max 1000 candles/request)
- Configurable interval (1m, 5m, 15m, 1h, etc.)
- Proxy rotation via ProxyManager (optional)

### 2.3 Proxy Manager (`proxies.py`)

The `ProxyManager` provides round-robin proxy rotation:
- Reads `proxies.txt` format: `ip:port:user:pass`
- Each `fetch()` call uses the next alive proxy in rotation
- Failed proxies are tracked: 3 failures → marked dead
- On failure, automatically retries with the next proxy
- Thread-safe (uses locks for concurrent access)
- Graceful fallback to direct connection if no proxies available

```python
pm = ProxyManager()
resp = pm.fetch("https://api.binance.com/api/v3/klines", params={...})
pm.status()  # {total: 500, alive: 497, dead: 3, index: 42}
```

---

## 3. Feature Layer (`research/features/`)

### 3.1 Technical Indicators (`indicators.py`)

All functions operate on pandas Series/DataFrames with vectorized operations:

| Function | Returns | Parameters |
|---|---|---|
| `sma(series, period)` | Moving average | period |
| `ema(series, period)` | Exponential MA | period, alpha |
| `rsi(series, period)` | 0-100 oscillator | period (default 14) |
| `atr(high, low, close, period)` | Avg True Range | period (default 14) |
| `true_range(high, low, close)` | True Range | — |
| `bollinger_bands(series, period, std)` | (mid, upper, lower) | period, std_dev |

### 3.2 ML Feature Matrix (`ml_features.py`)

`compute_features()` generates 35+ features from OHLCV:

**Price returns** (6 horizons):
- Simple returns: 1, 3, 5, 10, 20, 48 periods
- Log returns: same horizons
- Z-score of close (30, 100 periods)
- High-low range ratio

**Technical indicators:**
- RSI (7, 14, 21)
- MACD (line, signal, histogram)
- Bollinger %B and width
- ATR, ATR%, ATR ratio
- Volume ratio (20-period) and z-score (100-period)
- Price location within range (20, 50 periods)

**Time features:**
- Hour of day (sin/cos encoding)
- Day of week (sin/cos encoding)
- Weekend flag

All features use **only past data** — no look-ahead bias. Every rolling window uses `.shift()` or `.rolling()` with past values only.

`compute_target()` creates prediction targets:
- `binary`: 1 if forward return > 0, 0 otherwise
- `direction`: +1 (up), 0 (flat), -1 (down)
- `quantile`: Multi-class based on return distribution

`train_val_test_split()` provides chronological time-series splits (no random shuffling).

---

## 4. Backtest Engine (`research/backtest/`)

### 4.1 Architecture

```
VectorizedBacktest.run(signals)
        │
        ├── 1. Align signals to data index
        ├── 2. Shift signals by 1 (no look-ahead)
        ├── 3. Compute gross returns: position × close_return
        ├── 4. Subtract transaction costs: abs(Δposition) × cost_rate
        ├── 5. Cumprod to get equity curve
        └── 6. Extract trades (numba accelerated)
```

### 4.2 Equity Calculation

```
strategy_return[t] = position[t-1] × (close[t] / close[t-1] - 1)
cost[t] = |position[t-1] - position[t-2]| × (fee + slippage)
net_return[t] = strategy_return[t] - cost[t]
equity[t] = capital × ∏(1 + net_return)
```

All operations are vectorized (pandas/numpy) — no Python loops for the equity computation.

### 4.3 Batched Numba Kernels (`_batch.py`)

For Monte Carlo optimization, the engine switches to a fully compiled mode:

`single_backtest()` replaces the entire pandas pipeline with a single numba `@jit` pass:
- Takes raw numpy arrays: `close`, `signals`, `capital`, `fee_rate`
- Single pass over periods: price return, cost, equity, drawdown, trade tracking
- Returns 6 metrics: `final_equity, max_dd, sharpe, n_trades, win_rate, profit_factor`

`batch_ma_crossover()` extends this to 1000s of combos:
- Pre-computes SMA indicator banks (2D arrays)
- Uses `prange` to parallelize across all CPU cores
- Each combo is a lightweight array operation — no Python overhead

**Performance: 758 combos/second on 876K rows (540x vs sequential).**

### 4.4 Metrics (`metrics.py`)

| Metric | Definition | Target |
|---|---|---|
| Sharpe Ratio | mean(excess_return) / std × √periods | >1.0 |
| Sortino Ratio | mean(excess_return) / downside_std × √periods | >1.5 |
| Profit Factor | gross_profit / gross_loss | >1.8 |
| Calmar Ratio | CAGR / |max_drawdown| | >1.0 |
| Max Drawdown | peak-to-trough equity decline | <25% |
| CAGR | Compound Annual Growth Rate | >0% |
| Win Rate | winning trades / total trades | — |
| Expectancy | EV per trade in $ | >0 |

---

## 5. Strategy Layer (`research/strategies/`)

### 5.1 Base Strategy Interface (`base.py`)

Every strategy extends `BaseStrategy` and implements:

```python
class MyStrategy(BaseStrategy):
    DEFAULT_CONFIG = {"param1": 20, "param2": 0.5}

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Returns -1, 0, +1 (or continuous float for sizing)
        ...
```

### 5.2 Strategy Details

| Strategy | Module | Logic | State Machine |
|---|---|---|---|
| MA Crossover | `ma_crossover.py` | fast/slow MA cross, EMA/SMA | Vectorized (pure pandas) |
| Donchian Breakout | `breakout.py` | Price breaks N-period high/low | **Numba** |
| Mean Reversion | `mean_reversion.py` | RSI extremes or Bollinger squeeze | **Numba** |
| Volatility Breakout | `volatility_breakout.py` | ATR expansion → directional entry | **Numba** |
| Regime Detection | `regime.py` | SMA200 slope or ADX-based filter | Vectorized |
| Ensemble | `ensemble.py` | Rolling Sharpe-weighted combiner | Vectorized |

Strategies marked **Numba** use `_numba_ops.py` for the state machine loop, providing 50-100x speedup over Python iteration.

### 5.3 Numba-Accelerated State Machines (`_numba_ops.py`)

Four compiled functions replace Python for-loops:
- `donchian_breakout_numba()` — Donchian channel entry/exit
- `mean_reversion_rsi_numba()` — RSI oversold/overbought
- `mean_reversion_bollinger_numba()` — Bollinger Band touch
- `volatility_breakout_numba()` — ATR expansion + direction

Each takes raw numpy arrays, runs a state machine in `@jit(nopython=True)`, and returns a signal array. No Python objects inside the loop.

### 5.4 Ensemble System (`ensemble.py`)

Combines multiple strategy signals using:

- **Equal weight**: Simple average — robust, no look-back period needed
- **Sharpe weight**: Each strategy weighted by its recent rolling Sharpe ratio. Negative Sharpe → zero weight. Adapts to regime changes.
- **Sharpe rank**: Rank-based weighting (less extreme than raw Sharpe). The best strategy gets the highest weight but not all of it.

```python
combined = ensemble_signals(
    {"ma": macd_sig, "breakout": donchian_sig, "mr": meanrev_sig},
    data["close"],
    method="sharpe_weight",
    window=1440,  # ~5 days on 5m data
)
```

### 5.5 Regime Detection (`regime.py`)

Two modes:
- **SMA slope**: Classifies by price relative to long-term SMA. Default: bull > 105%, bear < 95%, sideways between.
- **ADX**: Uses Average Directional Index. ADX > 25 = trending (direction from +DI vs -DI), ADX < 25 = sideways.

`filter_by_regime()` zeros out signals incompatible with the current regime (longs in bear, shorts in bull, all allowed in sideways).

---

## 6. ML Pipeline (`research/ml/`)

### 6.1 Training (`trainer.py`)

```
Features (35+) → Chronological split → LightGBM → Probabilities → Signals
```

- **Model:** LightGBM binary classifier, `num_threads=32`
- **Training:** 220K rows (2017-2022), validation 47K (2023), test 47K (2024-2025)
- **Early stopping:** 50 rounds patience
- **Signal conversion:** Discrete threshold or continuous position sizing

Key finding: Pure ML on 5m OHLCV has very weak signal (logloss 0.6927 vs 0.6931 random). The model is best used as a small contributor in the ensemble.

### 6.2 Feature Importance (`importance.py`)

Generates reports ranking features by gain and split importance. Includes:
- Cumulative gain percentage
- Feature correlation matrix
- Ablation analysis (drop top N features, measure logloss impact)

Top features found: `close_z_100`, `atr_ratio`, `hour_sin/cos`, `atr_pct`.

---

## 7. Optimization Layer (`research/optimization/`)

### 7.1 Batched Monte Carlo (`monte_carlo.py`)

The primary optimization engine. Uses `prange` across all 32 threads:

```
Pre-compute indicator bank (2D array, once)         ← ~0.1s
    │
    ▼
for combo_idx in prange(10000):                      ← ~13s total
    signals = bank[fast] > bank[slow]                ← array op (ns)
    equity = compute_equity(close, signals)           ← single pass
    metrics[combo] = {sharpe, pf, dd, trades}         ← stats
```

Available search functions:
- `monte_carlo_ma(close, fast_range, slow_range)` — MA Crossover
- `monte_carlo_rsi(close, period_range, oversold_values, overbought_values)` — RSI Mean Rev

Results include sorted metrics and can be saved as text reports to `results/reports/`.

### 7.2 Walk-Forward Search (`search.py`)

Sequential random search with walk-forward validation:
- Train on 2017-2022, validate on 2023, test on 2024-2025
- Scores combos by validation Sharpe
- Generates optimization reports with top N results

### 7.3 Robustness Suite (`robustness.py`)

Three validation tests:

1. **Monte Carlo shuffle:** Shuffles trade outcomes 1000 times, compares actual Sharpe against random distribution. p-value < 0.05 = significant edge.

2. **Regime breakdown:** Runs backtest segregated by bull/bear/sideways regimes. Shows which market environments the strategy works in.

3. **Parameter stability:** Varies each parameter ±20% around the optimal value. Checks if metrics collapse (overfitting indicator).

---

## 8. Execution Layer (`research/execution/`)

### 8.1 Paper Trading (`paper.py`)

Generates signals from current market data and logs to `results/paper/paper_trades.jsonl`.

Pre-configured with optimized strategies:
- MA Crossover: (12, 26, EMA)
- Donchian Breakout: (30, 15)
- RSI Mean Reversion: (25, 30, 88) — **best found**

Run as a cron job: `python -c "from research.execution.paper import paper_trade_job; print(paper_trade_job())"`

Can be scheduled with a 5-minute interval for live signal generation.

---

## 9. Optimization History

### Numba Acceleration

| Component | Before | After | Speedup |
|---|---|---|---|
| Trade extraction | 50ms Python loop | 1ms numba | 50x |
| Donchian signal gen | 2s Python loop | 0.45s numba | 4x |
| RSI signal gen | 2s Python loop | 0.24s numba | 8x |
| Monte Carlo (10K combos) | ~6 hours | ~13s | 540x |

### Key Research Findings

| Finding | Sharpe | PF | DD | Notes |
|---|---|---|---|---|
| RSI Mean Rev (25, 30, 88) | +0.31 | 1.12 | -6.45% | First positive out-of-sample edge |
| MA Crossover (any params) | -2.8 to -10 | 0.65-0.84 | -100% | Unprofitable on 5m after fees |
| ML model (LightGBM) | -0.08 | 0.83 | -11% | Barely above random (0.6927 vs 0.6931) |

### Hardware Configuration

- **CPU:** 32 threads — numba prange uses all cores, ML training threads=32
- **RAM:** 96 GB — full dataset + indicator bank + ML training fits comfortably
- **SSD:** 7000 MB/s — Parquet I/O for 876K rows takes <0.5s

---

## 10. Project Structure

```
aaa-trade/
├── AGENTS.md                      Agent instructions (plan→todo→review→execute)
├── CLI.md                         CLI command reference
├── README.md                      Project overview
├── todo.md                        Task tracker
├── conftest.py                    Pytest configuration (adds project to sys.path)
├── .gitignore
├── proxies.txt                    Proxy list (gitignored, 500+ entries)
├── data/
│   ├── btc-yearly/                Raw JSON (9 files, 126 MB)
│   └── processed/                 Parquet (31 MB)
├── docs/                          Strategy guidelines, standards, whitepaper
├── research/
│   ├── data/loader.py             Loader + fetcher + cleaner
│   ├── data/proxies.py            Round-robin proxy manager
│   ├── features/indicators.py     Technical indicator library
│   ├── features/ml_features.py    ML feature matrix builder
│   ├── backtest/engine.py         Vectorized backtest engine
│   ├── backtest/_numba_ops.py     Numba trade extraction
│   ├── backtest/_batch.py         Batched numba kernels (Monte Carlo)
│   ├── backtest/metrics.py        15+ performance metrics
│   ├── strategies/                Strategy implementations
│   ├── ml/trainer.py              LightGBM pipeline
│   ├── ml/importance.py           Feature importance reports
│   ├── optimization/search.py     Walk-forward parameter search
│   ├── optimization/monte_carlo.py Batched Monte Carlo optimizer
│   ├── optimization/robustness.py Statistics + regime + stability tests
│   └── execution/paper.py         Paper trading pipeline
├── tests/                         106 tests across 16 files
└── results/
    ├── backtests/                 Per-run backtest reports
    ├── reports/                   Optimization & robustness reports
    └── paper/                     Paper trading logs
```
