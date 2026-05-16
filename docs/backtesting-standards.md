# Backtesting Standards

**Version:** 1.0  
**Applies to:** All strategy backtests within QuantumEdge  
**Goal:** Ensure every backtest is realistic, reproducible, and statistically meaningful.

---

## 1. Data Requirements

### 1.1 Source & Coverage
- **Primary source:** Binance (Spot and USDT Perpetual Futures)
- **Timeframe:** 5-minute candles (OHLCV + Volume)
- **Minimum period:** 5 years per asset (ideally 10+ years: 2016–2026)
- **Target assets (initial):** BTCUSDT, ETHUSDT
- **Expansion criteria:** >$50M daily volume, >2 years listing history

### 1.2 Quality Checks
Every dataset must pass these checks before use:
- No missing timestamps (enforce 5-min grid, flag gaps >1 candle)
- No zero-volume candles during exchange uptime
- No extreme outliers (price moves >50σ within 3 candles)
- Gap handling: forward-fill gaps ≤2 candles, mark larger gaps as NaN
- Timestamps in UTC, aligned to 5-min boundaries (00/05/10/...)

### 1.3 Storage Format
- Parquet (compressed) — preferred for performance
- CSV acceptable for ad-hoc / notebook work
- Index: datetime (UTC), sorted ascending
- Columns: `open, high, low, close, volume` (and `funding_rate` for futures)

---

## 2. Cost Modeling

All backtests MUST include realistic costs. No "zero-commission" runs accepted.

| Parameter | Spot | Futures |
|---|---|---|
| Maker fee | 0.075% | 0.02% |
| Taker fee | 0.075% | 0.04% |
| Slippage (entries) | 1 tick | 1 tick |
| Slippage (exits) | 1–2 ticks | 1–2 ticks |
| Funding rate | N/A | 8h rate (use historical) |

Slippage should increase during periods of low volume (lowest 10% percentile).

---

## 3. Backtesting Engine

### 3.1 Vectorized (Primary)
- Use vectorbt or a custom pandas/NumPy engine
- Prefer for: quick iteration, parameter sweeps, large-scale testing
- Must still model realistic fills (no cross-currency assumptions)

### 3.2 Event-Driven (Optional)
- Implement for: strategy-specific edge cases, limit orders, partial fills
- Use only when vectorized assumptions break down

---

## 4. Walk-Forward Methodology

Every strategy must pass walk-forward analysis before being considered viable.

1. **Split data** into sequential windows
2. **In-sample (IS):** first 70% of each window — optimize parameters
3. **Out-of-sample (OOS):** remaining 30% — validate
4. **Minimum windows:** 4 (covering bull, bear, sideways regimes)
5. **Anchor dates:** force crash periods into IS at least once (2018, 2020, 2022, 2025)
6. **Rolling validation:** retrain / re-optimize every 6 months

Pass criteria: OOS performance must be ≥70% of IS across all windows.

---

## 5. Key Performance Metrics

Every backtest report MUST include:

| Metric | Target | Definition |
|---|---|---|
| Profit Factor | >1.8 | Gross Profit / Gross Loss |
| Sharpe Ratio | >1.0 (annualized) | (Mean Return − Risk-Free) / Std Dev |
| Sortino Ratio | >1.5 (annualized) | Downside-deviation version of Sharpe |
| Calmar Ratio | >1.0 | CAGR / Max Drawdown |
| Max Drawdown | <25–30% | Peak-to-trough equity drop |
| Win Rate | — | Trades closed at profit / total trades |
| Avg RR | >1.5:1 | Avg Win / Avg Loss |
| Total Trades | >300 | Statistical confidence threshold |
| Expectancy | >0 | EV per dollar risked |

All ratios annualized using crypto market convention (365 days, 24h).

---

## 6. Statistical Validation

### 6.1 Required Tests
- **Monte Carlo simulation:** shuffle trade outcomes (1,000+ runs) → check if actual equity curve beats 95th percentile of random
- **Bootstrap test:** resample trade sequences with replacement (10,000 iterations) → confidence interval on Sharpe
- **t-test on daily returns:** H₀: mean return ≤ 0, reject at p < 0.05

### 6.2 Overfitting Prevention
- **Parameter stability test:** vary each parameter ±20% → strategy should not break
- **Cross-validation:** test across multiple uncorrelated assets
- **Regime robustness:** must survive bear, bull, and sideways markets independently

---

## 7. Minimum Acceptance Criteria

A strategy graduates from Phase 1 to Phase 2 only when ALL of:

- [ ] Positive EV in IS **and** OOS
- [ ] Profit Factor >1.5 in OOS
- [ ] Max DD <30% (all periods)
- [ ] >300 trades across full dataset
- [ ] Passes Monte Carlo at 95% confidence
- [ ] Survives 2018, 2020, 2022, 2025 crashes without >50% DD
- [ ] Parameter stability test passed

---

## 8. Reporting Format

Every backtest produces a standard report:
1. **Summary table** (all metrics above)
2. **Equity curve** (with DD overlay)
3. **Trade list** (timestamps, entries, exits, PnL)
4. **Monthly return heatmap**
5. **Distribution of returns** (histogram + QQ plot)
6. **Regime breakdown** (metrics per bull/bear/sideways)
7. **Walk-forward table** (IS vs OOS per window)
8. **Monte Carlo results** (p-value, confidence band)

Reports go into `results/backtests/<strategy_name>/<timestamp>/`.
