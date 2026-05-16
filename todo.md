# QuantumEdge — Project Todo

> Auto-generated from `docs/roadmap.md`. Check items off as we go.

---

## Phase 1: Foundation

- [x] Whitepaper & documentation
- [x] AGENTS.md optimized at root
- [ ] **P1-1:** Python venv setup + core dependencies (pandas, numpy, vectorbt, pytest, python-binance)
- [ ] **P1-2:** Data loader — read JSON yearly files, convert to Parquet, timestamp validation, basic cleaning
- [ ] **P1-3:** Exploratory data analysis notebook (volume patterns, volatility regimes, missing data)
- [ ] **P1-4:** Core backtest engine — vectorized signal → position → PnL pipeline
- [ ] **P1-5:** Metrics module (Sharpe, Sortino, Profit Factor, Max DD, Calmar, etc.)
- [ ] **P1-6:** Results directory structure (`results/backtests/`, `results/reports/`)

## Phase 2: Classical Strategies

- [ ] Moving average & breakout systems
- [ ] Mean reversion strategies
- [ ] Volatility-based strategies
- [ ] Regime detection (classical)

## Phase 3: Advanced Models

- [ ] Machine Learning models
- [ ] Feature importance analysis
- [ ] Ensemble system

## Phase 4: Validation & Deployment

- [ ] Rigorous statistical validation
- [ ] Paper trading
- [ ] Live trading (Python)
- [ ] Optional: Rust/Go migration
