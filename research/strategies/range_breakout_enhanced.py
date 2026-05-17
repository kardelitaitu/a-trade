"""
Range Breakout Limit — Enhanced with all 5 options.

All options disabled by default. Each can be toggled independently.

Options:
  O1: Regime filter (precomputed bull/bear/sideways array)
  O2: ATR-based limit instead of fixed limit_pct
  O3: Chandelier exit (trailing ATR from peak) instead of SMA exit
  O4: Multi-timeframe confirmation (precomputed 4h trend array)
  O5: Volume divergence (skip if pullback volume > breakout volume)
"""

import numpy as np
import pandas as pd
from numba import jit
from research.strategies.base import BaseStrategy
from research.strategies.regime import detect_regime_sma
from research.data.loader import resample_ohlcv


@jit(nopython=True)
def _enhanced_numba(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    atr: np.ndarray,
    regime: np.ndarray,
    trend_4h: np.ndarray,
    range_period: int,
    vol_sma_period: int,
    vol_mult: float,
    min_hold: int,
    limit_pct: float,
    limit_atr_mult: float,
    limit_max_bars: int,
    chandelier_mult: float,
    use_regime_filter: bool,
    use_atr_limit: bool,
    use_chandelier_exit: bool,
    use_mtf: bool,
    use_vol_divergence: bool,
    bull_regime: int,
) -> np.ndarray:
    """Enhanced state machine with all 5 optional filters."""
    n = len(close)
    signals = np.zeros(n, dtype=np.float64)
    if n == 0:
        return signals

    # Precompute running range
    highest = np.zeros(n)
    lowest_at = np.zeros(n)
    for i in range(n):
        highest[i] = high[max(0, i - range_period + 1):i + 1].max()
        lowest_at[i] = low[max(0, i - range_period + 1):i + 1].min()

    # Volume SMA buffer
    vol_cum = 0.0
    vol_buf = np.zeros(n)
    vol_sma_actual = max(vol_sma_period, 1)
    for i in range(n):
        vol_cum += volume[i]
        if i >= vol_sma_actual:
            vol_cum -= volume[i - vol_sma_actual]
        denom = i + 1 if i + 1 < vol_sma_actual else vol_sma_actual
        vol_buf[i] = vol_cum / max(denom, 1.0)

    # ATR SMA buffer (for chandelier exit)
    atr_sma = np.zeros(n)
    atr_cum = 0.0
    for i in range(n):
        atr_cum += atr[i]
        if i >= 14:
            atr_cum -= atr[i - 14]
        atr_sma[i] = atr_cum / (i + 1 if i + 1 < 14 else 14)

    in_position = False
    position_side = 0
    hold_since = 0
    entry_peak = 0.0

    limit_active = False
    limit_is_long = False
    limit_is_short = False
    limit_price = 0.0
    limit_bar = -1
    peak_since_breakout = 0.0
    breakout_vol = 0.0

    for i in range(1, n):
        if not in_position:
            if limit_active:
                if i - limit_bar > limit_max_bars:
                    limit_active = False

                # O5: Volume divergence check
                if use_vol_divergence and limit_active:
                    pullback_vol = 0.0
                    pullback_count = 0
                    for pb_i in range(limit_bar + 1, i + 1):
                        pullback_vol += volume[pb_i]
                        pullback_count += 1
                    if pullback_count > 0:
                        avg_pb_vol = pullback_vol / pullback_count
                        if avg_pb_vol > breakout_vol * 1.5:
                            limit_active = False  # Skip — too much selling

                if limit_is_long and low[i] <= limit_price:
                    signals[i] = 1.0
                    in_position = True
                    position_side = 1
                    hold_since = i
                    entry_peak = high[i]
                    limit_active = False
                elif limit_is_short and high[i] >= limit_price:
                    signals[i] = -1.0
                    in_position = True
                    position_side = -1
                    hold_since = i
                    entry_peak = low[i]
                    limit_active = False
                else:
                    if limit_is_long and high[i] > peak_since_breakout:
                        peak_since_breakout = high[i]
                        if use_atr_limit:
                            limit_price = peak_since_breakout - (atr_sma[i] * limit_atr_mult)
                        else:
                            limit_price = peak_since_breakout * (1.0 - limit_pct)
                    elif limit_is_short and low[i] < peak_since_breakout:
                        peak_since_breakout = low[i]
                        if use_atr_limit:
                            limit_price = peak_since_breakout + (atr_sma[i] * limit_atr_mult)
                        else:
                            limit_price = peak_since_breakout * (1.0 + limit_pct)

            if not limit_active and not in_position:
                long_signal = close[i] > highest[i - 1]
                short_signal = close[i] < lowest_at[i - 1]
                vol_ok = volume[i] >= vol_buf[i] * vol_mult if vol_mult > 0 else True

                # O1: Regime filter
                if use_regime_filter:
                    long_signal = long_signal and regime[i] == bull_regime
                    short_signal = short_signal and regime[i] == -bull_regime if bull_regime != 0 else short_signal

                # O4: MTF filter (only allow longs if 4h trend is up)
                if use_mtf:
                    long_signal = long_signal and trend_4h[i] > 0
                    short_signal = short_signal and trend_4h[i] < 0

                if long_signal and vol_ok:
                    limit_active = True
                    limit_is_long = True
                    limit_is_short = False
                    peak_since_breakout = high[i]
                    breakout_vol = volume[i]
                    if use_atr_limit:
                        limit_price = peak_since_breakout - (atr_sma[i] * limit_atr_mult)
                    else:
                        limit_price = peak_since_breakout * (1.0 - limit_pct)
                    limit_bar = i
                elif short_signal and vol_ok:
                    limit_active = True
                    limit_is_long = False
                    limit_is_short = True
                    peak_since_breakout = low[i]
                    breakout_vol = volume[i]
                    if use_atr_limit:
                        limit_price = peak_since_breakout + (atr_sma[i] * limit_atr_mult)
                    else:
                        limit_price = peak_since_breakout * (1.0 + limit_pct)
                    limit_bar = i
        else:
            held = i - hold_since
            if held < min_hold:
                signals[i] = position_side
                continue

            # Update peak for chandelier exit
            if position_side == 1 and high[i] > entry_peak:
                entry_peak = high[i]
            elif position_side == -1 and low[i] < entry_peak:
                entry_peak = low[i]

            if use_chandelier_exit:
                if position_side == 1 and close[i] < entry_peak - (chandelier_mult * atr_sma[i]):
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                elif position_side == -1 and close[i] > entry_peak + (chandelier_mult * atr_sma[i]):
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                else:
                    signals[i] = position_side
            else:
                ema_exit = close[max(0, i - range_period // 2 + 1):i + 1].mean()
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


def _bool_to_int(val):
    return 1 if val else 0


class RangeBreakoutEnhanced(BaseStrategy):
    """Range Breakout Limit with all 5 enhancement options.

    Config flags:
        use_regime_filter     : bool (default: False) — O1
        use_atr_limit         : bool (default: False) — O2 (replaces limit_pct)
        use_chandelier_exit   : bool (default: False) — O3 (replaces SMA exit)
        use_mtf               : bool (default: False) — O4
        use_vol_divergence    : bool (default: False) — O5

    Config params (when applicable):
        regime_period         : int (default: 50)
        limit_atr_mult        : float (default: 0.5) — ATR multiplier for limit
        chandelier_mult       : float (default: 3.0) — ATR multiplier for exit
    """

    DEFAULT_CONFIG = {
        "range_period": 32,
        "filter_volume": False,
        "filter_vol_sma_period": 20,
        "filter_vol_mult": 1.5,
        "min_hold": 2,
        "limit_pct": 0.008,
        "limit_max_bars": 3,
        "use_regime_filter": False,
        "use_atr_limit": False,
        "use_chandelier_exit": False,
        "use_mtf": False,
        "use_vol_divergence": False,
        "regime_period": 50,
        "limit_atr_mult": 0.5,
        "chandelier_mult": 3.0,
    }

    @property
    def name(self) -> str:
        return "Range Breakout Enhanced"

    @property
    def description(self) -> str:
        parts = [f"range={self.config['range_period']}"]
        if self.config["use_regime_filter"]:
            parts.append("regime")
        if self.config["use_atr_limit"]:
            parts.append("atr-limit")
        if self.config["use_chandelier_exit"]:
            parts.append("chandelier")
        if self.config["use_mtf"]:
            parts.append("mtf")
        if self.config["use_vol_divergence"]:
            parts.append("vol-div")
        return "Enhanced(" + ", ".join(parts) + ")"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].values.astype(np.float64)
        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        volume = data.get("volume", pd.Series(0, index=data.index)).values.astype(np.float64)

        # O2/O3: Precompute ATR
        atr_arr = np.zeros(len(close), dtype=np.float64)
        if self.config["use_atr_limit"] or self.config["use_chandelier_exit"]:
            from research.features.indicators import atr
            atr_series = atr(data["high"], data["low"], data["close"], 14)
            atr_arr = atr_series.values.astype(np.float64)

        # O1: Precompute regime where needed
        regime_arr = np.zeros(len(close), dtype=np.float64)
        if self.config["use_regime_filter"]:
            r = detect_regime_sma(data["close"], period=self.config["regime_period"])
            regime_arr = np.where(r == "bull", 1.0, np.where(r == "bear", -1.0, 0.0))

        # O4: Precompute 4h trend (resample, compute, back-fill)
        trend_arr = np.zeros(len(close), dtype=np.float64)
        if self.config["use_mtf"]:
            data_4h = resample_ohlcv(data, "4h")
            from research.features.indicators import sma
            sma_4h = sma(data_4h["close"], 20)
            trend_4h_s = data_4h["close"] > sma_4h
            trend_4h_s = trend_4h_s.reindex(data.index, method="ffill").fillna(1)
            trend_arr = trend_4h_s.values.astype(np.float64) * 2 - 1  # 1 = bull, -1 = bear

        signals = _enhanced_numba(
            close, high, low, volume, atr_arr, regime_arr, trend_arr,
            self.config["range_period"],
            self.config["filter_vol_sma_period"],
            self.config["filter_vol_mult"],
            self.config["min_hold"],
            self.config["limit_pct"],
            self.config["limit_atr_mult"],
            self.config["limit_max_bars"],
            self.config["chandelier_mult"],
            _bool_to_int(self.config["use_regime_filter"]),
            _bool_to_int(self.config["use_atr_limit"]),
            _bool_to_int(self.config["use_chandelier_exit"]),
            _bool_to_int(self.config["use_mtf"]),
            _bool_to_int(self.config["use_vol_divergence"]),
            1,  # bull_regime
        )

        return pd.Series(signals, index=data.index, dtype=float)
