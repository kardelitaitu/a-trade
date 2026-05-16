**Profitable trading, mathematically and statistically, boils down to having a positive expected value (EV) edge, rigorous risk management (especially proper position sizing), and enough statistical validity to trust that your edge persists.** Most retail traders lose money (typically 70-90%+ across studies and broker disclosures), largely because they trade without a verifiable edge, overleverage, or let emotions override math.

### 1. Expected Value (EV) — The Core Math
EV is the long-run average outcome per trade if you repeated it many times. Positive EV is required for profitability.

**Formula (per trade, simplified):**
EV = (Win Probability × Average Win Amount) − (Loss Probability × Average Loss Amount) − Costs (commissions, slippage, spreads)

- If EV > 0 → You have an edge (profitable over large samples).
- If EV ≤ 0 → You're expected to lose money regardless of win rate.

**Example:**
- Trader A: 70% win rate, avg win $100, avg loss $250 → EV = (0.7 × 100) - (0.3 × 250) = 70 - 75 = **-$5** (losing edge).
- Trader B: 40% win rate, avg win $400, avg loss $100 → EV = (0.4 × 400) - (0.6 × 100) = 160 - 60 = **+$100** (strong edge, even with more losses).

**Key insight:** Win rate alone is meaningless. Risk-reward ratio (e.g., 1:2 or better) often matters more. Always factor in real costs.

### 2. Risk Management and Position Sizing (Kelly Criterion)
Even with positive EV, poor sizing leads to ruin via drawdowns or variance.

**Core rules:**
- Never risk more than 1-2% of total capital on a single trade (conservative for most).
- Use stop-losses based on your strategy's volatility, not arbitrary levels.
- Diversify and avoid over-correlation.

**Kelly Criterion** for optimal (aggressive) sizing:
f = (bp − q) / b

Where:
- f = fraction of capital to risk
- p = win probability
- q = 1 - p (loss probability)
- b = odds (avg win / avg loss)

**Simplified version often used:** Kelly % = W − [(1−W)/R], where W = win rate, R = win/loss ratio.

**Example:** 60% win rate (W=0.6), 1.5:1 reward:risk (R=1.5) → Kelly ≈ 0.6 - (0.4/1.5) ≈ 33%. Many traders use **half-Kelly** (e.g., 16.5%) for lower volatility.

Kelly maximizes long-term geometric growth but can be volatile — overbetting (full Kelly with estimation errors) leads to large drawdowns.

### 3. Statistical Validity: Edge Must Be Real, Not Luck
- **Backtest properly**: Minimum 100-500+ trades (ideally 30+ per regime), out-of-sample testing, walk-forward optimization. Account for slippage, commissions, regime shifts (bull/bear markets).
- **Statistical significance**: Use t-tests, p-values (<0.05), bootstrapping, or Monte Carlo simulations to check if returns >0 or Sharpe ratio > random chance.
- **Survivorship & overfitting bias**: Test across multiple assets, time periods, and market conditions. A strategy that works only on 2020-2021 tech stocks is likely curve-fitted.
- **Sharpe/Sortino ratios**, maximum drawdown, and profit factor (gross profit / gross loss >1.5-2 ideally) help evaluate robustness.

Markets are close to efficient (especially liquid ones), so true edges are rare, small, and decay over time. Common sources: statistical arbitrage, market microstructure, behavioral anomalies, or execution advantages (for institutions).

### 4. Practical Mathematical Framework for Profitability
1. **Develop/test a strategy** with positive EV (backtest + forward test).
2. **Size positions** via Kelly/fixed-fractional (or volatility-based, e.g., ATR).
3. **Track metrics** rigorously: Use a trading journal with EV per trade, expectancy (EV per unit risked).
4. **Manage psychology** via rules-based execution — variance will cause losing streaks even with +EV.
5. **Compound** via positive expectancy + controlled risk. Geometric returns matter: small consistent edge + time beats home runs.
6. **Costs and taxes** erode edges — minimize them.
7. **Position in portfolio context**: Correlation, overall volatility, capital allocation.

**Simulation tip**: You can model this in code to see how win rate, R:R, and sizing interact over thousands of trades.

### Harsh Statistical Reality
- Retail failure rates are high because of overtrading, leverage chasing, no edge, and poor psychology.
- Beating the market consistently (after fees) is extremely difficult. Many "profitable" traders eventually blow up due to fat tails/black swans.
- Professional edges often come from scale, information, or technology—not retail chart patterns.

**Bottom line**: Profitable trading is a probability game of positive expectancy executed with iron discipline over thousands of trials. Start by proving (statistically) that your approach has +EV on historical + live data with realistic costs. If you can't quantify the edge, you're gambling. Most should consider passive indexing instead. 

Treat it like a business: measure everything, iterate on data, and respect the math of risk.