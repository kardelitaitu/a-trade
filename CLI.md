# QuantumEdge CLI Reference

## Setup & Environment

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/Scripts/activate

# Install dependencies
pip install pandas numpy python-binance vectorbt pytest pytest-cov \
            jupyter matplotlib seaborn pyarrow lightgbm
```

---

## Data Management

### Load raw JSON and save as Parquet
```bash
python -c "from research.data.loader import load_all, save_parquet; df = load_all(); save_parquet(df)"
```

### Load processed Parquet into a DataFrame
```python
from research.data.loader import load_parquet
data = load_parquet()                          # default: btcusdt_5m.parquet
data = load_parquet("eth_usdt_5m.parquet")     # specific asset
```

### Fetch historical data from Binance API
```bash
python -c "from research.data.loader import fetch_binance_klines; \
  df = fetch_binance_klines('BTCUSDT', '5m', '2024-01-01')"

# With proxy rotation (reads proxies.txt)
python -c "from research.data.loader import fetch_binance_klines; \
  df = fetch_binance_klines('ETHUSDT', '5m', '2024-01-01', use_proxies=True)"
```

---

## Running Strategies

### Single strategy backtest
```bash
python -c "
from research.data.loader import load_parquet
from research.strategies.mean_reversion import MeanReversion
from research.backtest.engine import VectorizedBacktest
from research.backtest.metrics import compute_metrics, format_metrics_report

data = load_parquet()
strat = MeanReversion({'mode': 'rsi', 'rsi_period': 25, 'rsi_oversold': 30, 'rsi_overbought': 88})
signals = strat.generate_signals(data)
result = VectorizedBacktest(data).run(signals)
metrics = compute_metrics(result.equity_curve, result.trades)
print(format_metrics_report(metrics))
"
```

### Available strategies
| Strategy | Import | Key params |
|---|---|---|
| MA Crossover | `from research.strategies.ma_crossover import MACrossover` | fast_period, slow_period, ma_type |
| Donchian Breakout | `from research.strategies.breakout import DonchianBreakout` | entry_period, exit_period |
| Mean Reversion | `from research.strategies.mean_reversion import MeanReversion` | mode (rsi/bollinger), rsi_period, oversold/overbought |
| Volatility Breakout | `from research.strategies.volatility_breakout import VolatilityBreakout` | atr_period, atr_multiplier |
| Regime Detection | `from research.strategies.regime import detect_regime_sma, filter_by_regime` | period, bull/bear threshold |

### Strategy with regime filter
```python
from research.strategies.regime import detect_regime_sma, filter_by_regime
regime = detect_regime_sma(data["close"])
filtered_signals = filter_by_regime(signals, regime)
```

### Ensemble (combine multiple strategies)
```python
from research.strategies.ensemble import ensemble_signals
combined = ensemble_signals(signal_dict, data["close"], method="sharpe_weight")
# methods: equal, sharpe_weight, sharpe_rank
```

---

## Backtesting

### VectorizedBacktest
```python
from research.backtest.engine import VectorizedBacktest, Trade, BacktestResult

# Default config
bt = VectorizedBacktest(data)
result = bt.run(signals)

# Custom fees / capital
bt = VectorizedBacktest(data, {"initial_capital": 100_000, "fee": 0.001, "slippage": 0.0002})
result = bt.run(signals)

# Access results
result.equity_curve       # pd.Series — portfolio value over time
result.trades             # list[Trade] — each closed trade
result.positions          # pd.Series — positions over time
result.signals            # pd.Series — input signals
result.config             # dict — config used
```

### Compute metrics
```python
from research.backtest.metrics import compute_metrics, format_metrics_report
metrics = compute_metrics(result.equity_curve, result.trades)
# Keys: sharpe_ratio, sortino_ratio, profit_factor, calmar_ratio,
#        max_drawdown_pct, cagr_pct, total_trades, win_rate_pct,
#        avg_rr, expectancy, total_fees, ...
print(format_metrics_report(metrics))
```

---

## Parameter Optimization

### Monte Carlo (batched, 758 combos/sec)
```bash
python -c "
from research.data.loader import load_parquet
from research.optimization.monte_carlo import monte_carlo_ma, save_mc_report
from pathlib import Path

data = load_parquet()
close = data['close'].values.astype(float)
result = monte_carlo_ma(close, (5, 50), (20, 100))
print(result.summary())
save_mc_report(result, Path('results/reports'))
"
```

### Monte Carlo for RSI
```bash
python -c "
from research.data.loader import load_parquet
from research.optimization.monte_carlo import monte_carlo_rsi, save_mc_report
data = load_parquet()
close = data['close'].values.astype(float)
result = monte_carlo_rsi(close, (7, 28), [20,25,30,35,40], [60,65,70,75,80])
print(result.summary())
"
```

### Walk-forward parameter search (sequential)
```bash
python -c "
from research.data.loader import load_parquet
from research.optimization.search import run_ma_crossover_search, run_mean_reversion_search, generate_optimization_report
from pathlib import Path

data = load_parquet()
results = run_mean_reversion_search(data, n=50)
generate_optimization_report(results, 'Mean Reversion', Path('results/reports'))
"
```

### Robustness tests
```python
from research.optimization.robustness import (
    monte_carlo_shuffle, regime_breakdown,
    parameter_stability, generate_robustness_report
)
mc = monte_carlo_shuffle(result.trades, n_simulations=1000)
rd = regime_breakdown(data, signals)
sd = parameter_stability(MeanReversion, params, data)
path = generate_robustness_report(mc, rd, sd, "Strategy Name", Path("results/reports"))
```

---

## ML Pipeline

### Train model
```bash
python -c "
from research.data.loader import load_parquet
from research.ml.trainer import run_ml_pipeline
from pathlib import Path
data = load_parquet().loc['2023-01-01':'2025-12-31']
result = run_ml_pipeline(data, horizon=12, continuous=True)
print(result['feature_importance'].head(10))
"
```

### Feature importance report
```bash
python -c "
from pathlib import Path
from research.ml.importance import load_model, compute_importance, generate_importance_report
model = load_model(Path('research/ml/models/lgb_model.txt'))
features = ['ret_1','ret_3','ret_5','rsi_14','macd','bb_pct_b','atr_pct','hour_sin','hour_cos','dow_sin']
importance = compute_importance(model, features)
generate_importance_report(importance, Path('results/reports'))
"
```

### Feature engineering
```python
from research.features.ml_features import compute_features, compute_target, train_val_test_split
features = compute_features(data)
target = compute_target(data, horizon=6, method="binary")
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(features, target)
```

---

## Technical Indicators

```python
from research.features.indicators import sma, ema, rsi, atr, bollinger_bands

sma_20 = sma(data["close"], 20)
ema_12 = ema(data["close"], 12)
rsi_14 = rsi(data["close"], 14)
atr_14 = atr(data["high"], data["low"], data["close"], 14)
mid, upper, lower = bollinger_bands(data["close"], 20, 2.0)
```

---

## Paper Trading

```bash
# Run paper trading (generates signals from latest data)
python -c "from research.execution.paper import paper_trade_job; print(paper_trade_job())"
```

---

## Proxy Manager

```python
from research.data.proxies import ProxyManager

pm = ProxyManager()
resp = pm.fetch("https://api.binance.com/api/v3/klines", params={"symbol": "BTCUSDT", "interval": "5m"})
print(pm.status())  # {total, alive, dead, index}
```

---

## Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook research/notebooks/

# Available notebooks:
#   01_eda_btcusdt.ipynb  — Exploratory Data Analysis
```

---

## Testing

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_backtest.py -v

# Run with coverage
python -m pytest --cov=research

# Run tests matching a keyword
python -m pytest -k "ma_crossover"

# Quick smoke test (skip slow tests)
python -m pytest -m "not slow"
```

**Total: 106 tests across 16 test files.**

---

## Reports & Results

Generated reports are saved to `results/reports/`:
```
results/reports/
├── feature_importance.txt
├── optimization_ma_crossover.txt
├── optimization_donchian_breakout.txt
├── optimization_mean_reversion.txt
├── robustness_mean_reversion_(rsi).txt
├── mc_ma_crossover.txt
└── mc_rsi_mean_reversion.txt
```

Paper trading logs go to `results/paper/paper_trades.jsonl`.
