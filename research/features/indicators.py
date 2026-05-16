"""
Technical indicators for QuantumEdge.

All functions operate on pandas Series/DataFrames and return pandas objects
aligned with the input. Vectorized / numba-accelerated where possible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int, alpha: float | None = None) -> pd.Series:
    """Exponential Moving Average."""
    if alpha is None:
        alpha = 2.0 / (period + 1)
    return series.ewm(alpha=alpha, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    Formula: RSI = 100 - (100 / (1 + RS))
    where RS = avg_gain / avg_loss over period.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    # Use Wilder's smoothing after initial SMA
    avg_gain = avg_gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = avg_loss.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # Pure uptrend: avg_loss = 0 → RSI = 100
    # Pure downtrend: avg_gain = 0 → RSI = 0  
    # Flat: both = 0 → RSI = 50
    rsi[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    rsi[(avg_gain == 0) & (avg_loss > 0)] = 0.0
    rsi[(avg_gain == 0) & (avg_loss == 0)] = 50.0
    return rsi


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range: max(high-low, |high-prev_close|, |low-prev_close|)."""
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    tr = true_range(high, low, close)
    # Initial SMA, then Wilder smoothing
    atr_ = tr.rolling(period, min_periods=period).mean()
    atr_ = atr_.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr_


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.

    Returns
    -------
    (middle, upper, lower) as pd.Series.
    """
    middle = sma(series, period)
    std = series.rolling(period).std(ddof=0)
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return middle, upper, lower
