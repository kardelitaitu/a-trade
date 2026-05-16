# QuantumEdge — Project Todo

> Auto-generated from `docs/roadmap.md`. Check items off as we go.

---

## Phase 1: Foundation

- [x] Whitepaper & documentation
- [x] AGENTS.md optimized at root
- [x] **P1-1:** Python venv setup + core dependencies (pandas, numpy, vectorbt, pytest, python-binance)
- [x] **P1-2:** Data loader — read JSON yearly files, convert to Parquet, timestamp validation, basic cleaning
- [x] **P1-3:** Exploratory data analysis notebook (volume patterns, volatility regimes, missing data)
- [x] **P1-4:** Core backtest engine — vectorized signal → position → PnL pipeline
- [x] **P1-5:** Metrics module (Sharpe, Sortino, Profit Factor, Max DD, Calmar, etc.)
- [x] **P1-6:** Results directory structure (`results/backtests/`, `results/reports/`)

## Phase 2: Classical Strategies

- [x] **P2-1:** Base Strategy abstract class + `research/strategies/` module structure
- [x] **P2-2:** MA Crossover strategy (SMA/EMA, configurable periods, trend filter)
- [x] **P2-3:** Breakout system (Donchian channel, N-period high/low breakout)
- [x] **P2-4:** Mean Reversion (RSI extremes / Bollinger Band squeeze)
- [x] **P2-5:** Volatility-based strategy (ATR breakout or volatility-targeting wrapper)
- [x] **P2-6:** Regime detection module (SMA200 slope / ADX, acts as strategy filter)

## Phase 3: Advanced Models

- [x] **P3-1:** ML Feature Engineering — feature matrix from OHLCV (price, tech, time features)
- [x] **P3-2:** ML Signal Model — LightGBM binary classifier, chrono split train/val/test
- [x] **P3-3:** Feature Importance Analysis — SHAP, ablation, report to results/reports/
- [x] **P3-4:** Ensemble System — rolling Sharpe-weighted combiner of P2 + P3 signals

## Phase 4: Validation & Deployment

- [x] **P4-1:** Systematic parameter optimization — random search + walk-forward validation
- [x] **P4-2:** Multi-asset validation on ETHUSDT
- [x] **P4-3:** Statistical robustness suite (Monte Carlo, regime breakdown, stability)
- [x] **P4-4:** Paper trading pipeline (live signals, no real money)
- [ ] **P4-5:** Live execution engine (Binance API, optional) — requires local Binance access

## Optimization: Hardware Utilization

- [x] **O-1:** Numba-accelerate backtest trade extraction (50-100x speedup)
- [x] **O-2:** Numba-accelerate strategy state machines (breakout, mean reversion, vol)
- [x] **O-3:** Increase ML training to 32 threads
- [x] **O-4:** Full-system stress test — all strategies × parameter sweep on 876K rows

## Feature: Indicator Registry + Tests

- [ ] **I-1:** Create `research/features/registry.py` — central catalog with `get_indicator()`, `list_indicators()`
- [ ] **I-2:** Implement + test **Price Trend** indicators (wma, hma, macd variants)
- [ ] **I-3:** Implement + test **Momentum** indicators (stoch, williams_r, cci, roc, momentum)
- [ ] **I-4:** Implement + test **Volatility** indicators (keltner, natr)
- [ ] **I-5:** Implement + test **Volume** indicators (obv, vwap, mfi, vol_delta, cmf)
- [ ] **I-6:** Implement + test **Price Structure** indicators (donchian, pivot)
- [ ] **I-7:** Update `research/strategies/factory.py` — generic strategy builder using registry
- [ ] **I-8:** Run full 106+ test suite — verify nothing broke
