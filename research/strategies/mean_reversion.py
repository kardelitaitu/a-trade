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

import pandas as pd

from research.strategies.base import BaseStrategy
from research.features.indicators import rsi, bollinger_bands


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
        rsi_period = self.config["rsi_period"]
        oversold = self.config["rsi_oversold"]
        overbought = self.config["rsi_overbought"]

        rsi_val = rsi(close, rsi_period)

        signals = pd.Series(0, index=close.index, dtype=float)
        in_long = False
        in_short = False

        for i in range(len(close)):
            r = rsi_val.iloc[i]
            if pd.isna(r):
                continue

            if in_long:
                if r >= 50:
                    in_long = False
                else:
                    signals.iloc[i] = 1
            elif in_short:
                if r <= 50:
                    in_short = False
                else:
                    signals.iloc[i] = -1

            if not (in_long or in_short):
                if r < oversold:
                    in_long = True
                    signals.iloc[i] = 1
                elif r > overbought:
                    in_short = True
                    signals.iloc[i] = -1

        return signals

    def _bollinger_signals(self, close: pd.Series) -> pd.Series:
        period = self.config["bb_period"]
        std = self.config["bb_std"]

        mid, upper, lower = bollinger_bands(close, period, std)

        signals = pd.Series(0, index=close.index, dtype=float)
        in_long = False
        in_short = False

        for i in range(len(close)):
            c = close.iloc[i]
            u = upper.iloc[i]
            l = lower.iloc[i]
            m = mid.iloc[i]

            if pd.isna(m):
                continue

            if in_long:
                if c >= m:
                    in_long = False
                else:
                    signals.iloc[i] = 1
            elif in_short:
                if c <= m:
                    in_short = False
                else:
                    signals.iloc[i] = -1

            if not (in_long or in_short):
                if c < l:
                    in_long = True
                    signals.iloc[i] = 1
                elif c > u:
                    in_short = True
                    signals.iloc[i] = -1

        return signals
