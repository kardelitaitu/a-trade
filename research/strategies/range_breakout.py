"""
Range Breakout Strategy.

Entry: price breaks outside of recent range (high/low of N periods).
Exit: price closes inside SMA trailing exit or opposite signal.
Optional volume filter to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from numba import jit

from research.strategies.base import BaseStrategy


@jit(nopython=True)
def _range_breakout_numba(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    range_period: int,
    exit_period: int,
    vol_filter_on: bool,
    vol_sma_period: int,
    vol_mult: float,
    min_hold: int,
) -> np.ndarray:
    """Numba state machine for Range Breakout."""
    n = len(close)
    signals = np.zeros(n, dtype=np.float64)
    if n == 0:
        return signals

    # Running range
    highest = np.zeros(n)
    lowest_at = np.zeros(n)

    # Volume SMA
    vol_cum = 0.0
    vol_buf = np.zeros(n)

    for i in range(n):
        highest[i] = high[max(0, i - range_period + 1):i + 1].max()
        lowest_at[i] = low[max(0, i - range_period + 1):i + 1].min()

    # Precompute volume SMA
    vol_sma = vol_sma_period if vol_filter_on else 1
    for i in range(n):
        vol_cum += volume[i]
        if i >= vol_sma:
            vol_cum -= volume[i - vol_sma]
        vol_buf[i] = vol_cum / (i + 1 if i + 1 < vol_sma else vol_sma)

    # State machine
    in_position = False
    position_side = 0  # 1 = long, -1 = short
    hold_since = 0

    for i in range(1, n):
        if not in_position:
            long_entry = close[i] > highest[i - 1]
            short_entry = close[i] < lowest_at[i - 1]
            vol_ok = (not vol_filter_on) or (volume[i] >= vol_buf[i] * vol_mult)

            if long_entry and vol_ok:
                signals[i] = 1.0
                in_position = True
                position_side = 1
                hold_since = i
            elif short_entry and vol_ok:
                signals[i] = -1.0
                in_position = True
                position_side = -1
                hold_since = i
        else:
            held = i - hold_since
            if held < min_hold:
                signals[i] = position_side
                continue

            # Exit conditions
            ema_exit = close[max(0, i - exit_period + 1):i + 1].mean()
            if position_side == 1 and close[i] < ema_exit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            elif position_side == -1 and close[i] > ema_exit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side
                # Check for reversal
                if close[i] > highest[i - 1] and position_side == -1:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                elif close[i] < lowest_at[i - 1] and position_side == 1:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0

    return signals


class RangeBreakout(BaseStrategy):
    """Range Breakout strategy.

    Enters when price breaks outside a recent range with optional volume
    confirmation. Exits when price crosses back inside a trailing SMA.

    Config:
        range_period : int  (default: 48)  — lookback for range detection
        exit_period  : int  (default: 20)  — SMA exit trailing period
        filter_volume    : bool (default: False) — enable volume filter
        filter_vol_mult  : float (default: 1.5) — volume multiplier threshold
        min_hold      : int  (default: 3) — minimum hold periods
    """

    DEFAULT_CONFIG = {
        "range_period": 48,
        "exit_period": 20,
        "filter_volume": False,
        "filter_vol_mult": 1.5,
        "min_hold": 3,
    }

    @property
    def name(self) -> str:
        return "Range Breakout"

    @property
    def description(self) -> str:
        return (
            f"RangeBreakout(range={self.config['range_period']}, "
            f"exit={self.config['exit_period']}, "
            f"vol_filter={self.config['filter_vol_mult'] if self.config['filter_volume'] else 'off'})"
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].values.astype(np.float64)
        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        volume = data.get("volume", pd.Series(0, index=data.index)).values.astype(np.float64)

        signals = _range_breakout_numba(
            close, high, low, volume,
            self.config["range_period"],
            self.config["exit_period"],
            self.config["filter_volume"],
            self.config.get("filter_vol_sma_period", 20),
            self.config["filter_vol_mult"],
            self.config["min_hold"],
        )

        return pd.Series(signals, index=data.index, dtype=float)
