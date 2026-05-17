"""
EMA Pullback Strategy — Trend Retracement Entry.

1. Bull market confirmed: SMA(50) > SMA(200)
2. Wait for pullback: price dips below EMA(20)
3. Enter long when price closes back above EMA(20)
4. Exit when close < SMA(10) or opposite signal
"""

import numpy as np
import pandas as pd
from numba import jit
from research.strategies.base import BaseStrategy


@jit(nopython=True)
def _ema_pullback_numba(
    close: np.ndarray,
    ema20: np.ndarray,
    sma50: np.ndarray,
    sma200: np.ndarray,
    exit_period: int,
    min_hold: int,
) -> np.ndarray:
    n = len(close)
    signals = np.zeros(n, dtype=np.float64)
    if n == 0:
        return signals

    in_position = False
    hold_since = 0
    pulled_back = False

    for i in range(1, n):
        bull_market = sma50[i] > sma200[i]

        if not in_position:
            if bull_market and not pulled_back:
                if close[i] < ema20[i]:
                    pulled_back = True
            elif bull_market and pulled_back:
                if close[i] > ema20[i]:
                    signals[i] = 1.0
                    in_position = True
                    hold_since = i
                    pulled_back = False

            if not bull_market:
                pulled_back = False
        else:
            held = i - hold_since
            if held < min_hold:
                signals[i] = 1.0
                continue

            sma_exit = close[max(0, i - exit_period + 1):i + 1].mean()
            if close[i] < sma_exit or not bull_market:
                signals[i] = 0.0
                in_position = False
            else:
                signals[i] = 1.0

    return signals


class EMAPullback(BaseStrategy):
    DEFAULT_CONFIG = {
        "exit_period": 10,
        "min_hold": 3,
        "trend_fast": 50,
        "trend_slow": 200,
        "pullback_ema": 20,
    }

    @property
    def name(self): return "EMA Pullback"

    @property
    def description(self):
        return f"EMAPullback(exit={self.config['exit_period']})"

    def generate_signals(self, data):
        from research.features.indicators import sma as sma_fn, ema as ema_fn
        close = data["close"].values.astype(np.float64)
        ema20 = ema_fn(data["close"], self.config["pullback_ema"]).values.astype(np.float64)
        sma50 = sma_fn(data["close"], self.config["trend_fast"]).values.astype(np.float64)
        sma200 = sma_fn(data["close"], self.config["trend_slow"]).values.astype(np.float64)
        sig = _ema_pullback_numba(close, ema20, sma50, sma200, self.config["exit_period"], self.config["min_hold"])
        return pd.Series(sig, index=data.index, dtype=float)
