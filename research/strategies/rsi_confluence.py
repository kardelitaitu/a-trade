"""
RSI Confluence Strategy.

1. Bull market: close > SMA(200)
2. Pullback: RSI(14) between 40-50 (oversold but not extreme)
3. Enter long when RSI crosses above 50
4. Exit when RSI > 70 (overbought) or close < SMA(10)
"""

import numpy as np
import pandas as pd
from numba import jit
from research.strategies.base import BaseStrategy


@jit(nopython=True)
def _rsi_confluence_numba(
    close: np.ndarray,
    rsi: np.ndarray,
    sma200: np.ndarray,
    exit_period: int,
    rsi_enter_min: float,
    rsi_enter_max: float,
    rsi_exit: float,
    min_hold: int,
) -> np.ndarray:
    n = len(close)
    signals = np.zeros(n, dtype=np.float64)
    if n == 0:
        return signals

    in_position = False
    hold_since = 0
    pullback_detected = False

    for i in range(1, n):
        bull_market = close[i] > sma200[i]

        if not in_position:
            if bull_market and not pullback_detected:
                if rsi_enter_min < rsi[i] < rsi_enter_max:
                    pullback_detected = True
            elif bull_market and pullback_detected:
                if rsi[i] > 50:
                    signals[i] = 1.0
                    in_position = True
                    hold_since = i
                    pullback_detected = False

            if not bull_market:
                pullback_detected = False
        else:
            held = i - hold_since
            if held < min_hold:
                signals[i] = 1.0
                continue

            sma_exit = close[max(0, i - exit_period + 1):i + 1].mean()
            if rsi[i] > rsi_exit or close[i] < sma_exit or not bull_market:
                signals[i] = 0.0
                in_position = False
            else:
                signals[i] = 1.0

    return signals


class RSIConfluence(BaseStrategy):
    DEFAULT_CONFIG = {
        "exit_period": 10,
        "rsi_period": 14,
        "rsi_enter_min": 40,
        "rsi_enter_max": 50,
        "rsi_exit": 70,
        "min_hold": 3,
        "trend_slow": 200,
    }

    @property
    def name(self): return "RSI Confluence"

    @property
    def description(self):
        return f"RSIConfluence(rsi={self.config['rsi_period']}, exit={self.config['exit_period']})"

    def generate_signals(self, data):
        from research.features.indicators import rsi as rsi_fn, sma as sma_fn
        close = data["close"].values.astype(np.float64)
        rsi_s = rsi_fn(data["close"], self.config["rsi_period"])
        rsi_arr = rsi_s.values.astype(np.float64)
        sma200_arr = sma_fn(data["close"], self.config["trend_slow"]).values.astype(np.float64)
        sig = _rsi_confluence_numba(close, rsi_arr, sma200_arr,
                                    self.config["exit_period"], self.config["rsi_enter_min"],
                                    self.config["rsi_enter_max"], self.config["rsi_exit"],
                                    self.config["min_hold"])
        return pd.Series(sig, index=data.index, dtype=float)
