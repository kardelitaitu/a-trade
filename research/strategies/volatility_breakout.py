"""
Volatility Breakout Strategy

Edge: Periods of volatility expansion often follow through with directional moves.
When ATR exceeds its rolling mean by a configurable multiple, enter in the
direction of the price move. Exit when volatility contracts back to normal.

This differs from Donchian breakout by responding to volatility expansion
rather than price level breakout — captures momentum after quiet periods.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research.strategies.base import BaseStrategy
from research.features.indicators import atr
from research.strategies._numba_ops import volatility_breakout_numba


class VolatilityBreakout(BaseStrategy):
    """
    Volatility Breakout strategy.

    Parameters
    ----------
    config : dict, optional
        - atr_period : int (default 14)
        - atr_multiplier : float (default 2.0). Entry when ATR > mult * ATR_mean.
        - atr_lookback : int (default 100). Period for rolling ATR mean.
        - min_hold : int (default 6). Minimum candles to hold a position (30 min).
        - filter_volume_pct : float, optional.
    """

    DEFAULT_CONFIG: dict[str, Any] = {
        "atr_period": 14,
        "atr_multiplier": 2.0,
        "atr_lookback": 100,
        "min_hold": 6,
        "filter_volume_pct": None,
    }

    @property
    def name(self) -> str:
        return "Volatility Breakout"

    @property
    def description(self) -> str:
        return (
            f"Enters on volatility expansion (ATR > {self.config['atr_multiplier']}x "
            f"rolling mean), exits when ATR normalizes. "
            f"Min hold: {self.config['min_hold']} candles."
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        atr_period = self.config["atr_period"]
        mult = self.config["atr_multiplier"]
        lookback = self.config["atr_lookback"]
        min_hold = self.config["min_hold"]

        atr_val = atr(data["high"], data["low"], close, atr_period)
        atr_mean = atr_val.rolling(lookback, min_periods=lookback // 2).mean()

        position_arr = volatility_breakout_numba(
            close.values.astype(np.float64),
            atr_val.values.astype(np.float64),
            atr_mean.values.astype(np.float64),
            float(mult),
            int(min_hold),
        )
        signals = pd.Series(position_arr, index=data.index)
        vol_pct = self.config.get("filter_volume_pct")
        if vol_pct is not None:
            vol_threshold = data["volume"].quantile(vol_pct)
            signals = signals.where(data["volume"] >= vol_threshold, 0)

        return signals
