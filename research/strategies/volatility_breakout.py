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

import pandas as pd

from research.strategies.base import BaseStrategy
from research.features.indicators import atr


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

        # Volatility expansion signal
        vol_expansion = atr_val > mult * atr_mean

        # Direction of price move
        price_change = close.diff()
        direction = price_change.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        signals = pd.Series(0, index=data.index, dtype=float)
        position = 0
        hold_count = 0

        for i in range(len(data)):
            if pd.isna(atr_val.iloc[i]) or pd.isna(atr_mean.iloc[i]):
                continue

            if position != 0:
                hold_count += 1

                # Exit when volatility normalizes (ATR back below mean)
                # or minimum hold reached and volatility contracting
                if not vol_expansion.iloc[i] and hold_count >= min_hold:
                    position = 0
                    hold_count = 0
                else:
                    signals.iloc[i] = position

            # Entry: volatility expansion + directional move
            if position == 0 and vol_expansion.iloc[i] and not pd.isna(price_change.iloc[i]):
                if price_change.iloc[i] > 0:
                    position = 1
                elif price_change.iloc[i] < 0:
                    position = -1
                hold_count = 0
                signals.iloc[i] = position

        # Volume filter
        vol_pct = self.config.get("filter_volume_pct")
        if vol_pct is not None:
            vol_threshold = data["volume"].quantile(vol_pct)
            signals = signals.where(data["volume"] >= vol_threshold, 0)

        return signals
