"""
Volatility Squeeze Strategy

Detects Bollinger Band squeezes inside Keltner Channels.
When BB width contracts inside Keltner → volatility squeeze (calm before storm).
Enter when price breaks out of the squeeze in either direction.
Exit when volatility normalizes or squeeze reverses.

Indicators: bb_pct_b, bb_width, keltner, atr
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research.strategies.base import BaseStrategy
from research.features.indicators import bollinger_bands
from research.features.registry import keltner, atr


class VolatilitySqueeze(BaseStrategy):
    """
    Volatility Squeeze strategy.

    Logic:
    1. Compute Bollinger Bands and Keltner Channels
    2. Squeeze = BB fully inside Keltner (BB_upper < Keltner_upper AND BB_lower > Keltner_lower)
    3. During squeeze: flat (waiting for breakout)
    4. Breakout: when price closes outside BB, enter in breakout direction
    5. Exit: when squeeze releases or opposite breakout

    Parameters
    ----------
    config : dict
        - bb_period : int (default 20)
        - bb_std : float (default 2.0)
        - keltner_period : int (default 20)
        - keltner_atr_mult : float (default 1.5)
        - squeeze_lookback : int (default 50) — periods to check for squeeze history
        - filter_volume_pct : float, optional
    """

    DEFAULT_CONFIG: dict[str, Any] = {
        "bb_period": 20,
        "bb_std": 2.0,
        "keltner_period": 20,
        "keltner_atr_mult": 1.5,
        "squeeze_lookback": 50,
        "min_hold": 3,
        "filter_volume_pct": None,
    }

    @property
    def name(self) -> str:
        return "Volatility Squeeze"

    @property
    def description(self) -> str:
        return (
            f"Volatility Squeeze (BB {self.config['bb_period']}/{self.config['bb_std']}σ, "
            f"Keltner {self.config['keltner_period']}/{self.config['keltner_atr_mult']}x ATR)"
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        bb_period = self.config["bb_period"]
        bb_std = self.config["bb_std"]
        kelt_period = self.config["keltner_period"]
        kelt_atr = self.config["keltner_atr_mult"]
        lookback = self.config["squeeze_lookback"]
        min_hold = self.config["min_hold"]

        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Compute indicators
        bb_mid, bb_upper, bb_lower = bollinger_bands(close, bb_period, bb_std)
        kelt_mid, kelt_upper, kelt_lower = keltner(high, low, close, kelt_period, kelt_atr)

        # Squeeze condition: BB fully inside Keltner
        squeeze = (bb_upper < kelt_upper) & (bb_lower > kelt_lower)

        # Breakout: price closes outside BB while squeeze was active
        # Long breakout: close > BB upper AND squeeze was active recently
        # Short breakout: close < BB lower AND squeeze was active recently
        squeeze_any = squeeze.rolling(lookback, min_periods=lookback // 2).max()

        # Build signals using state machine
        signals = pd.Series(0, index=data.index, dtype=float)
        in_long = False
        in_short = False
        hold_count = 0

        for i in range(len(data)):
            if pd.isna(bb_upper.iloc[i]) or pd.isna(kelt_upper.iloc[i]):
                continue

            if in_long:
                hold_count += 1
                # Exit: BB contracts back inside Keltner (squeeze returns)
                # or strong opposite signal
                if squeeze.iloc[i] and hold_count >= min_hold:
                    in_long = False
                    hold_count = 0
                else:
                    signals.iloc[i] = 1
            elif in_short:
                hold_count += 1
                if squeeze.iloc[i] and hold_count >= min_hold:
                    in_short = False
                    hold_count = 0
                else:
                    signals.iloc[i] = -1

            # Entry: squeeze was active recently + breakout
            if not (in_long or in_short):
                # Use vectorized squeeze_any instead of O(n) loop per candle
                recent_squeeze = squeeze_any.iloc[i]

                if recent_squeeze:
                    if close.iloc[i] > bb_upper.iloc[i] and not pd.isna(bb_upper.iloc[i]):
                        in_long = True
                        hold_count = 0
                        signals.iloc[i] = 1
                    elif close.iloc[i] < bb_lower.iloc[i] and not pd.isna(bb_lower.iloc[i]):
                        in_short = True
                        hold_count = 0
                        signals.iloc[i] = -1

        # Volume filter
        vol_pct = self.config.get("filter_volume_pct")
        if vol_pct is not None:
            vol_threshold = data["volume"].quantile(vol_pct)
            signals = signals.where(data["volume"] >= vol_threshold, 0)

        return signals