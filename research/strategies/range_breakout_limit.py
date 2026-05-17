"""
Range Breakout with Limit Retracement Entry.

1. Detect breakout: close > highest of N-period range (same as RangeBreakout)
2. Track the highest point reached since the breakout
3. Place limit order at `limit_pct % below the peak`
4. Enter when price pulls back to the limit level
5. Exit via trailing SMA
"""

import numpy as np
import pandas as pd
from numba import jit

from research.strategies.base import BaseStrategy


@jit(nopython=True)
def _range_breakout_limit_numba(
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
    limit_pct: float,
    limit_max_bars: int,
) -> np.ndarray:
    """Numba state machine for Range Breakout with Limit Retracement."""
    n = len(close)
    signals = np.zeros(n, dtype=np.float64)
    if n == 0:
        return signals

    # Precompute running range
    highest = np.zeros(n)
    lowest_at = np.zeros(n)

    # Volume SMA buffer
    vol_cum = 0.0
    vol_buf = np.zeros(n)

    for i in range(n):
        highest[i] = high[max(0, i - range_period + 1):i + 1].max()
        lowest_at[i] = low[max(0, i - range_period + 1):i + 1].min()

    vol_sma = vol_sma_period if vol_filter_on else 1
    for i in range(n):
        vol_cum += volume[i]
        if i >= vol_sma:
            vol_cum -= volume[i - vol_sma]
        vol_buf[i] = vol_cum / (i + 1 if i + 1 < vol_sma else vol_sma)

    # State
    in_position = False
    position_side = 0
    hold_since = 0

    # Limit order tracking
    limit_active = False
    limit_is_long = False
    limit_is_short = False
    limit_price = 0.0
    limit_bar = -1
    peak_since_breakout = 0.0

    for i in range(1, n):
        if not in_position:
            if limit_active:
                # Check if limit expired
                if i - limit_bar > limit_max_bars:
                    limit_active = False

                # Check limit fill (long)
                if limit_is_long and low[i] <= limit_price:
                    signals[i] = 1.0
                    in_position = True
                    position_side = 1
                    hold_since = i
                    limit_active = False
                elif limit_is_short and high[i] >= limit_price:
                    signals[i] = -1.0
                    in_position = True
                    position_side = -1
                    hold_since = i
                    limit_active = False
                else:
                    # Update peak if price goes even higher/lower
                    if limit_is_long:
                        if high[i] > peak_since_breakout:
                            peak_since_breakout = high[i]
                            limit_price = peak_since_breakout * (1.0 - limit_pct)
                    elif limit_is_short:
                        if low[i] < peak_since_breakout:
                            peak_since_breakout = low[i]
                            limit_price = peak_since_breakout * (1.0 + limit_pct)

            if not limit_active and not in_position:
                long_signal = close[i] > highest[i - 1]
                short_signal = close[i] < lowest_at[i - 1]
                vol_ok = (not vol_filter_on) or (volume[i] >= vol_buf[i] * vol_mult)

                if long_signal and vol_ok:
                    # Start limit order: wait for pullback from peak
                    limit_active = True
                    limit_is_long = True
                    limit_is_short = False
                    peak_since_breakout = high[i]
                    limit_price = peak_since_breakout * (1.0 - limit_pct)
                    limit_bar = i
                elif short_signal and vol_ok:
                    limit_active = True
                    limit_is_long = False
                    limit_is_short = True
                    peak_since_breakout = low[i]
                    limit_price = peak_since_breakout * (1.0 + limit_pct)
                    limit_bar = i
        else:
            # In position — manage exit
            held = i - hold_since
            if held < min_hold:
                signals[i] = position_side
                continue

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

    return signals


class RangeBreakoutLimit(BaseStrategy):
    """Range Breakout with Limit Retracement Entry.

    Config:
        range_period    : int  (default: 48) — lookback for range
        exit_period     : int  (default: 20) — SMA exit period
        filter_volume       : bool (default: False)
        filter_vol_mult     : float (default: 1.5)
        min_hold         : int  (default: 3)
        limit_pct        : float (default: 0.02) — pullback % from peak to enter
        limit_max_bars   : int  (default: 6) — max bars to wait for limit fill
    """

    DEFAULT_CONFIG = {
        "range_period": 48,
        "exit_period": 20,
        "filter_volume": False,
        "filter_vol_mult": 1.5,
        "min_hold": 3,
        "limit_pct": 0.02,
        "limit_max_bars": 6,
    }

    @property
    def name(self) -> str:
        return "Range Breakout Limit"

    @property
    def description(self) -> str:
        return (
            f"RangeBreakoutLimit(range={self.config['range_period']}, "
            f"exit={self.config['exit_period']}, "
            f"limit={self.config['limit_pct']*100:.1f}%)"
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].values.astype(np.float64)
        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        volume = data.get("volume", pd.Series(0, index=data.index)).values.astype(np.float64)

        signals = _range_breakout_limit_numba(
            close, high, low, volume,
            self.config["range_period"],
            self.config["exit_period"],
            self.config["filter_volume"],
            self.config.get("filter_vol_sma_period", 20),
            self.config["filter_vol_mult"],
            self.config["min_hold"],
            self.config["limit_pct"],
            self.config["limit_max_bars"],
        )

        return pd.Series(signals, index=data.index, dtype=float)
