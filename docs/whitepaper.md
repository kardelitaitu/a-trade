# QuantumEdge: A Data-Driven 5-Minute Cryptocurrency Trading System

**Whitepaper v0.2**  
**Date:** May 2026  
**Project:** Profitable algorithmic trading strategy using Binance 5m candle data (10+ years)  
**Goal:** Build a statistically robust, positive expectancy trading system with professional risk management.

## Abstract

QuantumEdge is a quantitative trading framework targeting Binance cryptocurrency markets on the 5-minute timeframe. The system leverages over 10 years of high-quality historical OHLCV data to discover and validate repeatable statistical edges.

The development follows a **Python-first prototyping approach** for maximum research speed and flexibility, with a clear migration path to Rust or Go for production execution if higher performance is required later.

Core principles:
- Positive Expected Value (EV)
- Rigorous statistical validation
- Strict risk management
- Reproducible and extensible codebase

## 1. Introduction & Motivation

Cryptocurrency markets offer unique opportunities due to 24/7 operation and high volatility, but the majority of traders lose money due to lack of edge and poor discipline.

QuantumEdge aims to solve this by building a fully systematic, mathematics-driven trading system. The project prioritizes **research velocity** during the prototyping phase while maintaining a professional architecture suitable for long-term deployment.

## 2. Technology Stack

**Primary Language (Prototyping & Research):** Python 3.11+  
**Reasons:**
- Best ecosystem for data analysis, backtesting, and machine learning
- Rapid iteration and experimentation
- Excellent AI-assisted development support
- Mature libraries (pandas, vectorbt, PyTorch, etc.)

**Future Migration (Production):** Rust or Go (optional)  
- Performance-critical components (live execution engine, high-frequency signal generation)
- Better concurrency and resource efficiency
- Long-running stability

This hybrid approach allows us to validate the strategy quickly in Python before optimizing only the necessary parts.

## 3. Data Description

- **Source:** Binance (Spot and/or USDT Perpetual Futures)
- **Timeframe:** 5-minute candles (OHLCV + Volume + optional funding rate)
- **Period:** ~10 years (2016 – 2026, ~1–1.2 million candles per major pair)
- **Target Assets:** BTCUSDT, ETHUSDT, and selected high-liquidity pairs
- **Data Quality:** Gap handling, outlier detection, realistic fee & slippage modeling

## 4. Mathematical Foundation

### 4.1 Expected Value
EV = (Win Probability × Avg Win) − (Loss Probability × Avg Loss) − Transaction Costs

All strategies must demonstrate **positive EV** across multiple market regimes.

### 4.2 Risk & Position Sizing
- 0.5–1% risk per trade (capital-based)
- Kelly Criterion (full / half / fractional)
- Volatility targeting (ATR-based)
- Portfolio-level drawdown controls

### 4.3 Key Performance Metrics
- Profit Factor (>1.8 target)
- Sharpe & Sortino Ratios
- Maximum Drawdown (<25–30% target)
- Calmar Ratio
- Statistical significance (Monte Carlo, bootstrapping, walk-forward)

## 5. Strategy Development Philosophy

We will develop strategies in three layers:

**Layer 1 (Classical):** Technical indicators, statistical arbitrage, breakout, mean reversion, volatility regimes.  
**Layer 2 (Advanced):** Machine Learning (feature importance, regime detection, gradient boosting).  
**Layer 3 (Ensemble):** Combination of uncorrelated signals with dynamic weighting.

All strategies will be:
- Fully rule-based or hybrid
- Heavily filtered (regime, liquidity, risk conditions)
- Tested across bull, bear, and sideways markets

## 6. Backtesting & Validation Standards

- Vectorized backtesting for speed (vectorbt / custom engine)
- Walk-forward optimization + out-of-sample testing
- Monte Carlo simulations and robustness tests
- Realistic costs (slippage, maker/taker fees, funding rates)
- Multi-asset and multi-regime validation
- Overfitting prevention through parameter stability and cross-validation

**Minimum Acceptance Criteria:**
- Positive EV in both in-sample and out-of-sample periods
- Survives major crashes (2018, 2020, 2022, 2025) with controlled risk
- Hundreds to thousands of trades for statistical confidence

## 7. Codebase Architecture

```plaintext
QuantumEdge/
├── research/                  # Python (Primary)
│   ├── data/                  # Fetchers, loaders, cleaning
│   ├── features/              # Indicators, ML features
│   ├── strategies/            # Strategy classes
│   ├── backtest/              # Engine, metrics, visualization
│   ├── optimization/          # Walk-forward, hyperparam tuning
│   ├── ml/                    # Models and regime detection
│   ├── risk/                  # Kelly, sizing, stops
│   └── notebooks/             # Research & analysis
│
├── execution/                 # Future: Rust/Go live engine
├── shared/                    # Data models & communication
├── config/
├── results/                   # Reports, equity curves, logs
└── main.py