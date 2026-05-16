"""
MA Crossover Strategy

Edge: Trend-following via fast/slow moving average crossover.
Fast MA crossing above slow MA = uptrend (long).
Fast MA crossing below slow MA = downtrend (short).

Optional filters:
- Volume filter: skip trades when volume is in bottom N percentile
- Volatility filter: skip trades during extreme volatility (>3σ ATR)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from research.strategies.base import BaseStrategy
from research.features.indicators import sma, ema, atr


class MACrossover(BaseStrategy):
    """
    Moving Average Crossover strategy.

    Parameters
    ----------
    config : dict, optional
        - fast_period : int (default 12)
        - slow_period : int (default 26)
        - ma_type : str, 'sma' or 'ema' (default 'ema')
        - filter_volume_pct : float, optional (0-1). Skip when volume below this percentile.
        - filter_atr_mult : float, optional. Skip when ATR > N * rolling_mean(ATR).
        - atr_period : int (default 14). Period for ATR calculation.
    """

    DEFAULT_CONFIG: dict[str, Any] = {
        "fast_period": 12,
        "slow_period": 26,
        "ma_type": "ema",
        "filter_volume_pct": None,   # e.g. 0.1 = skip bottom 10% volume
        "filter_atr_mult": None,     # e.g. 3.0 = skip ATR > 3x mean
        "atr_period": 14,
    }

    @property
    def name(self) -> str:
        return "MA Crossover"

    @property
    def description(self) -> str:
        return (
            f"Trend-following strategy using {self.config['ma_type'].upper()} "
            f"({self.config['fast_period']}/{self.config['slow_period']}) crossover."
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        fast_p = self.config["fast_period"]
        slow_p = self.config["slow_period"]
        ma_type = self.config["ma_type"]

        close = data["close"]

        # Compute fast and slow MAs
        if ma_type == "sma":
            fast_ma = sma(close, fast_p)
            slow_ma = sma(close, slow_p)
        else:
            fast_ma = ema(close, fast_p)
            slow_ma = ema(close, slow_p)

        # Raw crossover signals
        signals = pd.Series(0, index=data.index)
        signals[fast_ma > slow_ma] = 1
        signals[fast_ma < slow_ma] = -1

        # Apply volume filter
        vol_pct = self.config.get("filter_volume_pct")
        if vol_pct is not None:
            vol_threshold = data["volume"].quantile(vol_pct)
            low_vol = data["volume"] < vol_threshold
            signals[low_vol] = 0

        # Apply volatility filter
        atr_mult = self.config.get("filter_atr_mult")
        if atr_mult is not None:
            atr_val = atr(data["high"], data["low"], data["close"], self.config["atr_period"])
            atr_mean = atr_val.rolling(100, min_periods=50).mean()
            extreme_vol = atr_val > atr_mult * atr_mean
            signals[extreme_vol] = 0

        return signals
