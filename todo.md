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

- [ ] Machine Learning models
- [ ] Feature importance analysis
- [ ] Ensemble system

## Phase 4: Validation & Deployment

- [ ] Rigorous statistical validation
- [ ] Paper trading
- [ ] Live trading (Python)
- [ ] Optional: Rust/Go migration
