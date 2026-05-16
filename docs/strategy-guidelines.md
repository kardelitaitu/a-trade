# Strategy Development Guidelines

**Version:** 1.0  
**Applies to:** All strategies built within QuantumEdge  
**Goal:** Standardize how strategies are designed, implemented, tested, and documented.

---

## 1. Strategy Development Process

```
Hypothesis → Prototype → Backtest → Validate → Document → Graduate
```

Each stage must be completed before moving to the next.

### Stage 1: Hypothesis
- State the edge clearly in one sentence
  - *Good:* "BTC tends to revert to its 20-period moving average after a 2σ deviation on the 5m chart during low-volatility regimes."
  - *Bad:* "I think BTC might go up sometimes."
- Identify the market regime(s) where the hypothesis applies
- Define the conditions under which the hypothesis fails

### Stage 2: Prototype
- Implement as a Python class in `research/strategies/`
- Use vectorized logic (pandas/numpy) for speed
- Test on 1 month of data to catch bugs before full backtest

### Stage 3: Backtest
- Run full backtest per `backtesting-standards.md`
- Generate standard report to `results/backtests/`

### Stage 4: Validate
- Walk-forward analysis
- Monte Carlo / bootstrap
- Multi-asset testing

### Stage 5: Document
- Fill out strategy card (Section 4 below)
- Commit README for the strategy

### Stage 6: Graduate
- Passes minimum acceptance criteria → moves to next phase
- Otherwise → revisit hypothesis or discard

---

## 2. Strategy Interface

Every strategy class MUST implement this interface:

```python
class BaseStrategy:
    def __init__(self, config: dict):
        """Initialize with config parameters."""
        self.config = config

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals.

        Args:
            data: OHLCV DataFrame with columns [open, high, low, close, volume]

        Returns:
            signal: -1 (short), 0 (neutral), +1 (long)
        """
        ...

    def set_params(self, **kwargs):
        """Override strategy parameters."""
        ...

    @property
    def name(self) -> str:
        """Human-readable strategy name."""
        ...

    @property
    def description(self) -> str:
        """One-paragraph description of the strategy logic."""
        ...
```

### Signal Convention
| Value | Meaning |
|---|---|
| -1 | Short / Sell |
| 0  | Neutral / No position |
| +1 | Long / Buy |

No open positions when signal = 0.

---

## 3. Development Layers

Per whitepaper Section 5, strategies are developed in three layers:

### Layer 1: Classical (Phase 2)
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR)
- Breakout systems (channel break, volatility break)
- Mean reversion (Bollinger squeeze, RSI extremes)
- Volatility regimes (ATR percentile-based filtering)

### Layer 2: Machine Learning (Phase 3)
- Feature engineering from technical indicators (no raw prices as features)
- Gradient boosting (XGBoost, LightGBM, CatBoost)
- Regime detection (HMM, clustering on volatility/volume)
- Feature importance analysis required before deployment

### Layer 3: Ensemble (Phase 3)
- Combine uncorrelated signals (correlation <0.3 between strategies)
- Dynamic weighting based on recent regime performance
- Minimum 3 independent strategies in ensemble

---

## 4. Strategy Card Template

Every strategy must have a strategy card in its docstring or an adjacent README:

```markdown
## Strategy: [Name]

**Layer:** Classical / ML / Ensemble  
**Type:** Trend / Mean Reversion / Breakout / Volatility / Hybrid  
**Timeframe:** 5m  

### Edge Statement
One-sentence description of the statistical edge.

### Entry Conditions
- Condition 1: ...
- Condition 2: ...
- Filter: ...

### Exit Conditions
- Stop loss: ...
- Take profit: ...
- Trailing stop: ...

### Risk Parameters
- Position size: ... (% of capital or Kelly fraction)
- Max concurrent positions: ...
- Max risk per trade: ...

### Filters
- Regime filter: ...
- Volume filter: ...
- Time filter: ...

### Known Failure Modes
- Strong trends with no retracement
- Choppy range-bound markets (for breakout systems)
- Low-volume periods / news events
```

---

## 5. Parameter Naming Convention

| Prefix | Domain | Example |
|---|---|---|
| `atr_` | ATR-based | `atr_period`, `atr_multiplier` |
| `ma_` | Moving average | `ma_fast`, `ma_slow`, `ma_type` |
| `bb_` | Bollinger Bands | `bb_period`, `bb_std` |
| `rsi_` | RSI | `rsi_period`, `rsi_oversold`, `rsi_overbought` |
| `vol_` | Volume | `vol_ma_period`, `vol_threshold` |
| `regime_` | Regime detection | `regime_lookback`, `regime_threshold` |
| `risk_` | Risk management | `risk_per_trade`, `risk_max_concurrent` |
| `stop_` | Stop loss / take profit | `stop_atr`, `stop_profit` |

All parameters live in a flat config dict or YAML file, no nested configs.

---

## 6. Filters (Mandatory)

Every strategy MUST apply at least one filter before entering a trade:

| Filter | Purpose |
|---|---|
| **Regime** | Only trade when market regime matches strategy design |
| **Volume** | Skip low-liquidity periods (bottom 10% volume percentile) |
| **Volatility** | Skip when ATR is extreme (>3σ from rolling mean) |
| **Time** | Exclude low-activity hours (optional, depending on strategy) |
| **Correlation** | Do not open correlated positions across multiple assets |

---

## 7. Testing Checklist

Before marking a strategy as "ready for next phase":

- [ ] Interface compliance (implements `generate_signals`, `set_params`, etc.)
- [ ] No hardcoded magic numbers (all in config)
- [ ] Vectorized implementation (no Python loops over candles)
- [ ] Filters active (>=1 filter applied)
- [ ] Strategy card documented
- [ ] Backtest report generated
- [ ] Walk-forward completed (IS vs OOS)
- [ ] Parameter stability test passed
- [ ] Multi-asset test passed (min 2 assets)

---

## 8. What NOT To Do

- Do NOT use future data in signal generation (look-ahead bias)
- Do NOT optimize on the full dataset (always IS/OOS split)
- Do NOT change strategy rules after seeing OOS results (cherry-picking)
- Do NOT use price itself as a feature (use returns, ratios, or normalized values)
- Do NOT skip cost modeling "for simplicity"
- Do NOT keep a strategy that failed validation — discard and learn

---

## 9. Versioning

- Strategies get version numbers: `v1.0`, `v1.1`, etc.
- Version bumps on material logic changes
- Each version has its own strategy card and backtest report
- Old versions are archived, not deleted
