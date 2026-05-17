"""
Volatility Expansion Strategy.

1. Compute ATR(14) and its 20-period SMA
2. Enter long when ATR > SMA(ATR, 20) * mult (volatility expansion)
   AND price breaks short-term range (close > highest high of N periods)
3. Exit when close < SMA(10) or ATR contracts below SMA
"""

import numpy as np
import pandas as pd
from numba import jit
from research.strategies.base import BaseStrategy


@jit(nopython=True)
def _vol_expansion_numba(
    close: np.ndarray,
    high: np.ndarray,
    atr: np.ndarray,
    atr_sma: np.ndarray,
    range_period: int,
    exit_period: int,
    atr_mult: float,
    min_hold: int,
) -> np.ndarray:
    n = len(close)
    signals = np.zeros(n, dtype=np.float64)
    if n == 0:
        return signals

    highest = np.zeros(n)
    for i in range(n):
        highest[i] = high[max(0, i - range_period + 1):i + 1].max()

    in_position = False
    hold_since = 0

    for i in range(1, n):
        if not in_position:
            vol_expanding = atr_sma[i] > 0 and atr[i] > atr_sma[i] * atr_mult
            if vol_expanding and close[i] > highest[i - 1]:
                signals[i] = 1.0
                in_position = True
                hold_since = i
        else:
            held = i - hold_since
            if held < min_hold:
                signals[i] = 1.0
                continue

            sma_exit = close[max(0, i - exit_period + 1):i + 1].mean()
            vol_contracted = atr[i] < atr_sma[i]
            if close[i] < sma_exit or vol_contracted:
                signals[i] = 0.0
                in_position = False
            else:
                signals[i] = 1.0

    return signals


class VolatilityExpansion(BaseStrategy):
    DEFAULT_CONFIG = {
        "range_period": 12,
        "exit_period": 10,
        "atr_mult": 1.2,
        "min_hold": 3,
        "atr_period": 14,
        "atr_sma_period": 20,
    }

    @property
    def name(self): return "Volatility Expansion"

    @property
    def description(self):
        return f"VolExpansion(range={self.config['range_period']}, mult={self.config['atr_mult']})"

    def generate_signals(self, data):
        from research.features.indicators import atr as atr_fn, sma as sma_fn
        close = data["close"].values.astype(np.float64)
        high = data["high"].values.astype(np.float64)
        atr_s = atr_fn(data["high"], data["low"], data["close"], self.config["atr_period"])
        atr_arr = atr_s.values.astype(np.float64)
        atr_sma_arr = sma_fn(atr_s, self.config["atr_sma_period"]).values.astype(np.float64)
        sig = _vol_expansion_numba(close, high, atr_arr, atr_sma_arr,
                                   self.config["range_period"], self.config["exit_period"],
                                   self.config["atr_mult"], self.config["min_hold"])
        return pd.Series(sig, index=data.index, dtype=float)
