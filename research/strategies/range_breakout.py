"""
Range Breakout Strategy with optional Pullback Entry.

Entry: price breaks outside of recent range (high/low of N periods).
  With pullback: marks breakout, waits for retest, then enters on bounce.
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
    pullback_pct: float,
    pullback_max_bars: int,
) -> np.ndarray:
    """Numba state machine for Range Breakout with optional pullback."""
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
    position_side = 0
    hold_since = 0

    # Pullback tracking
    breakout_bar = -1
    breakout_level = 0.0
    breakout_is_long = False
    breakout_is_short = False

    for i in range(1, n):
        if not in_position:
            # ── Pullback mode: waiting for retest ──
            if breakout_bar >= 0:
                bars_since = i - breakout_bar
                if bars_since > pullback_max_bars:
                    # Timeout — cancel signal
                    breakout_bar = -1
                    breakout_level = 0.0
                    breakout_is_long = False
                    breakout_is_short = False
                else:
                    # Check for pullback
                    if breakout_is_long:
                        if low[i] <= breakout_level * (1.0 - pullback_pct):
                            # Pullback occurred — enter on next bounce
                            if close[i] > high[i - 1]:
                                signals[i] = 1.0
                                in_position = True
                                position_side = 1
                                hold_since = i
                                breakout_bar = -1
                                breakout_level = 0.0
                                breakout_is_long = False
                                breakout_is_short = False
                    elif breakout_is_short:
                        if high[i] >= breakout_level * (1.0 + pullback_pct):
                            if close[i] < low[i - 1]:
                                signals[i] = -1.0
                                in_position = True
                                position_side = -1
                                hold_since = i
                                breakout_bar = -1
                                breakout_level = 0.0
                                breakout_is_long = False
                                breakout_is_short = False

            # ── Fresh breakout detection ──
            if breakout_bar < 0 and not in_position:
                long_signal = close[i] > highest[i - 1]
                short_signal = close[i] < lowest_at[i - 1]
                vol_ok = (not vol_filter_on) or (volume[i] >= vol_buf[i] * vol_mult)

                if long_signal and vol_ok:
                    if pullback_pct > 0:
                        # Start pullback wait — reference breakout price
                        breakout_bar = i
                        breakout_level = close[i]
                        breakout_is_long = True
                        breakout_is_short = False
                    else:
                        # Enter immediately (no pullback)
                        signals[i] = 1.0
                        in_position = True
                        position_side = 1
                        hold_since = i
                elif short_signal and vol_ok:
                    if pullback_pct > 0:
                        breakout_bar = i
                        breakout_level = close[i]
                        breakout_is_long = False
                        breakout_is_short = True
                    else:
                        signals[i] = -1.0
                        in_position = True
                        position_side = -1
                        hold_since = i
        else:
            # ── In position — manage exit ──
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
                # Reversal exit
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
    """Range Breakout strategy with optional pullback entry.

    Config:
        range_period    : int  (default: 48)  — lookback for range detection
        exit_period     : int  (default: 20)  — SMA exit trailing period
        filter_volume       : bool (default: False) — enable volume filter
        filter_vol_mult     : float (default: 1.5) — volume multiplier
        min_hold         : int  (default: 3) — min hold periods
        pullback_pct     : float (default: 0.0) — pullback to wait for (0 = off)
        pullback_max_bars: int  (default: 6) — max bars to wait for pullback
    """

    DEFAULT_CONFIG = {
        "range_period": 48,
        "exit_period": 20,
        "filter_volume": False,
        "filter_vol_mult": 1.5,
        "min_hold": 3,
        "pullback_pct": 0.0,
        "pullback_max_bars": 6,
    }

    @property
    def name(self) -> str:
        return "Range Breakout"

    @property
    def description(self) -> str:
        pb = f", pullback={self.config['pullback_pct']*100:.1f}%" if self.config["pullback_pct"] > 0 else ""
        return (
            f"RangeBreakout(range={self.config['range_period']}, "
            f"exit={self.config['exit_period']}{pb})"
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
            self.config["pullback_pct"],
            self.config["pullback_max_bars"],
        )

        return pd.Series(signals, index=data.index, dtype=float)
