"""
Mean Reversion Strategy

Two modes:
- rsi: Long when RSI < oversold threshold, short when RSI > overbought,
       exit when RSI crosses back through 50.
- bollinger: Long when close < lower band, short when close > upper band,
             exit when close crosses back to the middle band.

Edge: Short-term price extremes tend to revert to the mean.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research.strategies.base import BaseStrategy
from research.features.indicators import rsi, bollinger_bands
from research.strategies._numba_ops import (
    mean_reversion_rsi_numba,
    mean_reversion_bollinger_numba,
)


class MeanReversion(BaseStrategy):
    """
    Mean Reversion strategy.

    Parameters
    ----------
    config : dict, optional
        - mode : str, 'rsi' or 'bollinger' (default 'rsi')
        - rsi_period : int (default 14)
        - rsi_oversold : float (default 30). Entry long when RSI below this.
        - rsi_overbought : float (default 70). Entry short when RSI above this.
        - bb_period : int (default 20)
        - bb_std : float (default 2.0)
        - filter_volume_pct : float, optional
    """

    DEFAULT_CONFIG: dict[str, Any] = {
        "mode": "rsi",
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "bb_period": 20,
        "bb_std": 2.0,
        "filter_volume_pct": None,
    }

    @property
    def name(self) -> str:
        return f"Mean Reversion ({self.config['mode'].upper()})"

    @property
    def description(self) -> str:
        if self.config["mode"] == "rsi":
            return (
                f"RSI mean reversion: long when RSI({self.config['rsi_period']}) "
                f"< {self.config['rsi_oversold']}, short when > {self.config['rsi_overbought']}."
            )
        return (
            f"Bollinger mean reversion: long at lower band "
            f"({self.config['bb_period']}, {self.config['bb_std']}σ), "
            f"short at upper band."
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        mode = self.config["mode"]
        close = data["close"]

        if mode == "rsi":
            signals = self._rsi_signals(close)
        else:
            signals = self._bollinger_signals(close)

        # Volume filter
        vol_pct = self.config.get("filter_volume_pct")
        if vol_pct is not None:
            vol_threshold = data["volume"].quantile(vol_pct)
            signals = signals.where(data["volume"] >= vol_threshold, 0)

        return signals

    def _rsi_signals(self, close: pd.Series) -> pd.Series:
        oversold = self.config["rsi_oversold"]
        overbought = self.config["rsi_overbought"]
        rsi_val = rsi(close, self.config["rsi_period"])

        arr = mean_reversion_rsi_numba(
            rsi_val.values.astype(np.float64),
            float(oversold),
            float(overbought),
        )
        return pd.Series(arr, index=close.index)

    def _bollinger_signals(self, close: pd.Series) -> pd.Series:
        period = self.config["bb_period"]
        std = self.config["bb_std"]
        mid, upper, lower = bollinger_bands(close, period, std)

        arr = mean_reversion_bollinger_numba(
            close.values.astype(np.float64),
            upper.values.astype(np.float64),
            lower.values.astype(np.float64),
            mid.values.astype(np.float64),
        )
        return pd.Series(arr, index=close.index)
