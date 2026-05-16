"""
Numba-accelerated strategy signal generation.

Each function replaces a Python state-machine loop with a compiled
numba loop for 50-100x speedup on large datasets.
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def donchian_breakout_numba(
    close: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    exit_upper: np.ndarray,
    exit_lower: np.ndarray,
) -> np.ndarray:
    """
    Donchian breakout state machine.
    Returns array of signals (-1, 0, +1).
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.float64)
    in_long = False
    in_short = False

    for i in range(n):
        c = close[i]
        u = upper[i]
        l = lower[i]

        if np.isnan(u) or np.isnan(l):
            continue

        if in_long:
            if c < exit_lower[i]:
                in_long = False
            else:
                signals[i] = 1.0
        elif in_short:
            if c > exit_upper[i]:
                in_short = False
            else:
                signals[i] = -1.0

        if not (in_long or in_short):
            if c > u:
                in_long = True
                signals[i] = 1.0
            elif c < l:
                in_short = True
                signals[i] = -1.0

    return signals


@jit(nopython=True)
def mean_reversion_rsi_numba(
    rsi_values: np.ndarray,
    oversold: float,
    overbought: float,
) -> np.ndarray:
    """
    RSI mean reversion state machine.
    Returns array of signals (-1, 0, +1).
    """
    n = len(rsi_values)
    signals = np.zeros(n, dtype=np.float64)
    in_long = False
    in_short = False

    for i in range(n):
        r = rsi_values[i]
        if np.isnan(r):
            continue

        if in_long:
            if r >= 50.0:
                in_long = False
            else:
                signals[i] = 1.0
        elif in_short:
            if r <= 50.0:
                in_short = False
            else:
                signals[i] = -1.0

        if not (in_long or in_short):
            if r < oversold:
                in_long = True
                signals[i] = 1.0
            elif r > overbought:
                in_short = True
                signals[i] = -1.0

    return signals


@jit(nopython=True)
def mean_reversion_bollinger_numba(
    close: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    middle: np.ndarray,
) -> np.ndarray:
    """
    Bollinger Band mean reversion state machine.
    Returns array of signals (-1, 0, +1).
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.float64)
    in_long = False
    in_short = False

    for i in range(n):
        c = close[i]
        u = upper[i]
        l = lower[i]
        m = middle[i]

        if np.isnan(m):
            continue

        if in_long:
            if c >= m:
                in_long = False
            else:
                signals[i] = 1.0
        elif in_short:
            if c <= m:
                in_short = False
            else:
                signals[i] = -1.0

        if not (in_long or in_short):
            if c < l:
                in_long = True
                signals[i] = 1.0
            elif c > u:
                in_short = True
                signals[i] = -1.0

    return signals


@jit(nopython=True)
def volatility_breakout_numba(
    close: np.ndarray,
    atr_val: np.ndarray,
    atr_mean: np.ndarray,
    multiplier: float,
    min_hold: int,
) -> np.ndarray:
    """
    Volatility breakout state machine.
    Returns array of signals (-1, 0, +1).
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.float64)
    position = 0.0
    hold_count = 0

    for i in range(n):
        a = atr_val[i]
        m = atr_mean[i]

        if np.isnan(a) or np.isnan(m):
            continue

        vol_expansion = a > multiplier * m

        if position != 0.0:
            hold_count += 1
            if not vol_expansion and hold_count >= min_hold:
                position = 0.0
                hold_count = 0
            else:
                signals[i] = position

        if position == 0.0:
            if i > 0 and vol_expansion:
                price_change = close[i] - close[i - 1]
                if price_change > 0:
                    position = 1.0
                elif price_change < 0:
                    position = -1.0
                hold_count = 0
                if position != 0.0:
                    signals[i] = position

    return signals
