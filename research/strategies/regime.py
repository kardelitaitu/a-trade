"""
Regime Detection Module

Identifies market regime (bull/bear/sideways) and can filter strategy
signals to only trade during compatible regimes.

Two modes:
- sma_slope: Price relative to long-term SMA (simple, fast)
- adx: ADX-based trending vs ranging detection
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

from research.features.indicators import ema

# Type for regime labels
Regime = Literal["bull", "bear", "sideways"]


def detect_regime_sma(
    close: pd.Series,
    period: int = 200,
    bull_threshold: float = 1.05,
    bear_threshold: float = 0.95,
) -> pd.Series:
    """
    Classify regime by price relative to long-term SMA.

    Parameters
    ----------
    close : pd.Series
    period : int
        SMA period (default 200, corresponds to ~7 days on 5m data).
    bull_threshold : float
        Price > SMA * threshold = bull.
    bear_threshold : float
        Price < SMA * threshold = bear.

    Returns
    -------
    pd.Series of str: 'bull', 'bear', 'sideways'
    """
    sma_val = close.rolling(period, min_periods=period // 2).mean()

    regime = pd.Series("sideways", index=close.index)
    regime[close > sma_val * bull_threshold] = "bull"
    regime[close < sma_val * bear_threshold] = "bear"
    return regime


def detect_regime_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    trending_threshold: float = 25.0,
) -> pd.Series:
    """
    Classify regime using ADX (Average Directional Index).

    ADX > threshold = trending (bull or bear based on +DI vs -DI).
    ADX < threshold = sideways.

    Parameters
    ----------
    high, low, close : pd.Series
    period : int
        ADX period (default 14).
    trending_threshold : float
        ADX above this = trending market.

    Returns
    -------
    pd.Series of str: 'bull', 'bear', 'sideways'
    """
    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(0.0, index=close.index)
    minus_dm = pd.Series(0.0, index=close.index)

    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    # Smoothed with Wilder's method
    tr_smooth = ema(tr, period)
    plus_di = 100 * ema(plus_dm, period) / tr_smooth.replace(0, np.nan)
    minus_di = 100 * ema(minus_dm, period) / tr_smooth.replace(0, np.nan)

    # DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = ema(dx, period)

    regime = pd.Series("sideways", index=close.index)
    trending = adx_val > trending_threshold
    regime[trending & (plus_di > minus_di)] = "bull"
    regime[trending & (minus_di > plus_di)] = "bear"

    return regime


def filter_by_regime(
    signals: pd.Series,
    regime: pd.Series,
    allowed_regimes: Optional[list[str]] = None,
) -> pd.Series:
    """
    Zero out signals that don't match the current regime.

    By default:
    - Long signals (+1) are only allowed in 'bull' regime
    - Short signals (-1) are only allowed in 'bear' regime
    - All signals are allowed in 'sideways' (no trend = any direction)

    Parameters
    ----------
    signals : pd.Series (-1, 0, +1)
    regime : pd.Series of str ('bull', 'bear', 'sideways')
    allowed_regimes : list[str], optional
        Override default behavior. E.g. ['bull', 'sideways'] to
        allow longs in bull and sideways but never short.

    Returns
    -------
    pd.Series with incompatible signals zeroed.
    """
    filtered = signals.copy()

    if allowed_regimes is not None:
        # Custom mode: only allow non-zero signals in listed regimes
        filtered[~regime.isin(allowed_regimes)] = 0
    else:
        # Default: long = bull only, short = bear only, sideways = either
        filtered[(signals == 1) & (regime != "bull") & (regime != "sideways")] = 0
        filtered[(signals == -1) & (regime != "bear") & (regime != "sideways")] = 0

    return filtered
