"""
Donchian Breakout Strategy

Edge: Momentum/trend following via Donchian channel breakout.
When price breaks above the N-period highest high, go long.
When price breaks below the N-period lowest low, go short.
Exit when price crosses back through the opposite channel boundary.

Optional filters:
- Volume filter (skip low volume periods)
- Volatility filter (skip extreme volatility)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research.strategies.base import BaseStrategy
from research.features.indicators import atr
from research.strategies._numba_ops import donchian_breakout_numba


class DonchianBreakout(BaseStrategy):
    """
    Donchian Channel Breakout strategy.

    Parameters
    ----------
    config : dict, optional
        - entry_period : int (default 20). Lookback for breakout detection.
        - exit_period : int (default 10). Lookback for exit signal.
        - filter_volume_pct : float, optional. Skip when volume below this percentile.
        - filter_atr_mult : float, optional. Skip when ATR > N * rolling_mean(ATR).
        - atr_period : int (default 14).
    """

    DEFAULT_CONFIG: dict[str, Any] = {
        "entry_period": 20,
        "exit_period": 10,
        "filter_volume_pct": None,
        "filter_atr_mult": None,
        "atr_period": 14,
    }

    @property
    def name(self) -> str:
        return "Donchian Breakout"

    @property
    def description(self) -> str:
        return (
            f"Momentum breakout using {self.config['entry_period']}-period "
            f"Donchian channel with {self.config['exit_period']}-period exit."
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        entry_p = self.config["entry_period"]
        exit_p = self.config["exit_period"]

        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Donchian channel
        upper = high.rolling(entry_p).max()
        lower = low.rolling(entry_p).min()
        exit_upper = high.rolling(exit_p).max()
        exit_lower = low.rolling(exit_p).min()

        # Entry signals
        long_entry = close > upper.shift(1)
        short_entry = close < lower.shift(1)

        # Exit signals
        long_exit = close < exit_lower.shift(1)
        short_exit = close > exit_upper.shift(1)

        # Build position using numba-accelerated state machine
        close_arr = close.values.astype(np.float64)
        upper_arr = upper.shift(1).fillna(0).values.astype(np.float64)
        lower_arr = lower.shift(1).fillna(0).values.astype(np.float64)
        exit_upper_arr = exit_upper.shift(1).fillna(0).values.astype(np.float64)
        exit_lower_arr = exit_lower.shift(1).fillna(0).values.astype(np.float64)

        position_arr = donchian_breakout_numba(
            close_arr, upper_arr, lower_arr, exit_upper_arr, exit_lower_arr,
        )
        signals = pd.Series(position_arr, index=data.index)

        # Apply volume filter
        vol_pct = self.config.get("filter_volume_pct")
        if vol_pct is not None:
            vol_threshold = data["volume"].quantile(vol_pct)
            signals = signals.where(data["volume"] >= vol_threshold, 0)

        # Apply volatility filter
        atr_mult = self.config.get("filter_atr_mult")
        if atr_mult is not None:
            atr_val = atr(data["high"], data["low"], data["close"], self.config["atr_period"])
            atr_mean = atr_val.rolling(100, min_periods=50).mean()
            signals = signals.where(atr_val <= atr_mult * atr_mean, 0)

        return signals
