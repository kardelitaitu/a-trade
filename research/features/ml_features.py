"""
ML Feature Engineering for QuantumEdge.

Computes a broad feature matrix from OHLCV data for ML model training.
All features use only past data — no look-ahead bias.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from research.features.indicators import rsi, atr, bollinger_bands


def compute_features(
    data: pd.DataFrame,
    include_time_features: bool = True,
    add_returns: bool = True,
    add_technical: bool = True,
) -> pd.DataFrame:
    """
    Compute a feature matrix from OHLCV data.

    Parameters
    ----------
    data : pd.DataFrame
        OHLCV data with columns [open, high, low, close, volume]
        and a datetime index (5-min).
    include_time_features : bool
        Add cyclical time features (hour, day of week).
    add_returns : bool
        Add price return features at various horizons.
    add_technical : bool
        Add technical indicator features.

    Returns
    -------
    pd.DataFrame
        Feature matrix with datetime index. Feature columns are named
        with descriptive prefixes.
    """
    close = data["close"]
    high = data["high"]
    low = data["low"]
    vol = data["volume"]
    features = pd.DataFrame(index=data.index)

    # ------------------------------------------------------------------
    # Price returns
    # ------------------------------------------------------------------
    if add_returns:
        for period in [1, 3, 5, 10, 20, 48]:  # 5min, 15min, 25min, 50min, 100min, 4h
            ret = close.pct_change(period)
            features[f"ret_{period}"] = ret
            # Log returns for better normality
            log_ret = np.log(close / close.shift(period))
            features[f"log_ret_{period}"] = log_ret

        # Rolling z-score of close (30, 100 periods)
        for period in [30, 100]:
            roll_mean = close.rolling(period).mean()
            roll_std = close.rolling(period).std(ddof=0)
            features[f"close_z_{period}"] = (close - roll_mean) / roll_std.replace(0, np.nan)

        # High-low range ratio
        features["hl_range"] = (high - low) / close

    # ------------------------------------------------------------------
    # Technical indicators
    # ------------------------------------------------------------------
    if add_technical:
        # RSI
        for period in [7, 14, 21]:
            features[f"rsi_{period}"] = rsi(close, period)

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        features["macd"] = macd_line
        features["macd_signal"] = macd_signal
        features["macd_hist"] = macd_line - macd_signal

        # Bollinger Bands %B
        mid, upper, lower = bollinger_bands(close, 20, 2.0)
        bb_width = upper - lower
        features["bb_pct_b"] = (close - lower) / bb_width.replace(0, np.nan)
        features["bb_width"] = bb_width / close * 100  # % width

        # ATR normalized
        atr_14 = atr(high, low, close, 14)
        features["atr_14"] = atr_14
        features["atr_pct"] = atr_14 / close * 100
        atr_50 = atr(high, low, close, 50)
        features["atr_ratio"] = atr_14 / atr_50.replace(0, np.nan)

        # Volume features
        vol_ma_20 = vol.rolling(20).mean()
        features["vol_ratio_20"] = vol / vol_ma_20.replace(0, np.nan)
        vol_z = (vol - vol.rolling(100).mean()) / vol.rolling(100).std(ddof=0).replace(0, np.nan)
        features["vol_z_100"] = vol_z

        # Price location within recent range
        for period in [20, 50]:
            hh = high.rolling(period).max()
            ll = low.rolling(period).min()
            features[f"price_loc_{period}"] = (close - ll) / (hh - ll).replace(0, np.nan)

    # ------------------------------------------------------------------
    # Time features (cyclical encoding)
    # ------------------------------------------------------------------
    if include_time_features:
        idx = data.index
        hour = idx.hour
        minute = idx.minute
        dow = idx.dayofweek

        # Hour as cyclical (sin/cos)
        hour_angle = 2 * np.pi * (hour + minute / 60) / 24
        features["hour_sin"] = np.sin(hour_angle)
        features["hour_cos"] = np.cos(hour_angle)

        # Day of week as cyclical
        dow_angle = 2 * np.pi * dow / 7
        features["dow_sin"] = np.sin(dow_angle)
        features["dow_cos"] = np.cos(dow_angle)

        # Weekend flag
        features["is_weekend"] = (dow >= 5).astype(float)

    # ------------------------------------------------------------------
    # Clean up: drop rows with NaN (warm-up period)
    # ------------------------------------------------------------------
    # Don't drop here — leave it to the caller to handle
    # (different models need different warm-up periods)

    return features


def compute_target(
    data: pd.DataFrame,
    horizon: int = 6,
    method: str = "binary",
) -> pd.Series:
    """
    Compute target labels for supervised learning.

    Parameters
    ----------
    data : pd.DataFrame
        OHLCV data.
    horizon : int
        Number of 5-min periods forward to predict (default 6 = 30 min).
    method : str
        'binary': 1 if future return > 0, 0 if <= 0.
        'direction': 1 (up), -1 (down), 0 (flat within threshold).
        'quantile': Multi-class based on return quantiles.

    Returns
    -------
    pd.Series
        Target labels aligned with data index. Last `horizon` rows are NaN.
    """
    future_close = data["close"].shift(-horizon)
    future_return = (future_close / data["close"]) - 1

    if method == "binary":
        target = (future_return > 0).astype(int)
        # Ensure last horizon rows are NaN (no future data)
        target.iloc[-horizon:] = np.nan if horizon > 0 else target
    elif method == "direction":
        threshold = 0.001  # 0.1%
        target = pd.Series(0, index=data.index)
        target[future_return > threshold] = 1
        target[future_return < -threshold] = -1
    elif method == "quantile":
        # 5 classes based on return quintiles
        target = pd.qcut(
            future_return.dropna(),
            q=5,
            labels=[-2, -1, 0, 1, 2],
            duplicates="drop",
        ).astype(int)
        # Reindex to original
        target = target.reindex(data.index)
    else:
        raise ValueError(f"Unknown method: {method}")

    return target


def train_val_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    val_split: float = 0.15,
    test_split: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Chronological train/val/test split (no shuffling — time series).

    Parameters
    ----------
    features : pd.DataFrame
    target : pd.Series
    val_split : float
        Fraction for validation (from end).
    test_split : float
        Fraction for test (from end, after val).

    Returns
    -------
    (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    n = len(features)
    val_start = int(n * (1 - val_split - test_split))
    test_start = int(n * (1 - test_split))

    X_train = features.iloc[:val_start]
    y_train = target.iloc[:val_start]

    X_val = features.iloc[val_start:test_start]
    y_val = target.iloc[val_start:test_start]

    X_test = features.iloc[test_start:]
    y_test = target.iloc[test_start:]

    return X_train, X_val, X_test, y_train, y_val, y_test
