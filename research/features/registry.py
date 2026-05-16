"""
Feature Registry for QuantumEdge.

Central catalog of all technical indicators. Every indicator is registered
with a name, function reference, parameter metadata, and description.

Usage:
    from research.features.registry import get_indicator, list_indicators

    fn = get_indicator("rsi")
    rsi_values = fn(close, period=14)

    list_indicators()
    # rsi  — Relative Strength Index (period: int)
    # atr  — Average True Range (period: int)
    # ...
"""

from __future__ import annotations

from typing import Any, Callable

from research.features.indicators import (
    # Existing from Phase 1
    sma, ema, rsi, atr, bollinger_bands, true_range,
    # New - will be added incrementally
)


# ---------------------------------------------------------------------------
# New indicators to add to indicators.py
# ---------------------------------------------------------------------------

def wma(series, period):
    """Weighted Moving Average — more weight to recent prices."""
    import numpy as np
    weights = np.arange(1, period + 1)
    def _wm(arr):
        return np.dot(arr, weights) / weights.sum()
    return series.rolling(period).apply(_wm, raw=True)


def hma(series, period):
    """Hull Moving Average — reduced lag vs SMA/EMA."""
    import numpy as np
    half = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    wma_half = wma(series, half)
    wma_full = wma(series, period)
    raw = 2 * wma_half - wma_full
    return wma(raw, sqrt_period)


def macd_line(close, fast=12, slow=26):
    """MACD Line: fast EMA - slow EMA."""
    return ema(close, fast) - ema(close, slow)


def macd_signal(close, fast=12, slow=26, signal=9):
    """MACD Signal Line: EMA of MACD Line."""
    return ema(macd_line(close, fast, slow), signal)


def macd_hist(close, fast=12, slow=26, signal=9):
    """MACD Histogram: MACD Line - Signal Line."""
    return macd_line(close, fast, slow) - macd_signal(close, fast, slow, signal)


def stoch(high, low, close, k_period=14, d_period=3):
    """Stochastic Oscillator %K and %D. Returns (%K, %D)."""
    import numpy as np
    low_min = low.rolling(k_period).min()
    high_max = high.rolling(k_period).max()
    k = 100 * (close - low_min) / (high_max - low_min).replace(0, np.nan)
    d = sma(k, d_period)
    return k, d


def williams_r(high, low, close, period=14):
    """Williams %R. Returns 0 to -100."""
    import numpy as np
    high_max = high.rolling(period).max()
    low_min = low.rolling(period).min()
    return -100 * (high_max - close) / (high_max - low_min).replace(0, np.nan)


def cci(high, low, close, period=20):
    """Commodity Channel Index."""
    import numpy as np
    tp = (high + low + close) / 3
    sma_tp = sma(tp, period)
    mad = (tp - sma_tp).abs().rolling(period).mean()
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))


def roc(close, period=10):
    """Rate of Change (%) = (close / close[N] - 1) * 100."""
    return (close / close.shift(period) - 1) * 100


def momentum(close, period=10):
    """Raw momentum = close - close[N]."""
    return close - close.shift(period)


def keltner(high, low, close, period=20, atr_mult=2.0):
    """Keltner Channel. Returns (middle, upper, lower)."""
    middle = ema(close, period)
    k_atr = atr(high, low, close, period)
    upper = middle + atr_mult * k_atr
    lower = middle - atr_mult * k_atr
    return middle, upper, lower


def natr(high, low, close, period=14):
    """Normalized ATR = ATR / close * 100."""
    return atr(high, low, close, period) / close * 100


def obv(close, volume):
    """On-Balance Volume — cumulative volume signed by price direction."""
    import numpy as np
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).fillna(0).cumsum()


def vwap(high, low, close, volume, period=20):
    """Volume-Weighted Average Price over N periods."""
    tp = (high + low + close) / 3
    pv = tp * volume
    return pv.rolling(period).sum() / volume.rolling(period).sum().replace(0, float("nan"))


def mfi(high, low, close, volume, period=14):
    """Money Flow Index — RSI-like but volume-weighted."""
    import numpy as np
    tp = (high + low + close) / 3
    raw_mf = tp * volume
    direction = np.sign(tp.diff().fillna(0))
    pos_mf = (direction > 0) * raw_mf
    neg_mf = (direction < 0) * raw_mf
    mfr = pos_mf.rolling(period).sum() / neg_mf.rolling(period).sum().replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def vol_delta(close, open_price, volume, period=1):
    """Volume Delta = (close - open) * volume. Smoothed by period SMA.
    
    Positive = buying pressure. Negative = selling pressure.
    """
    raw = (close - open_price) * volume
    return sma(raw, period) if period > 1 else raw


def cmf(high, low, close, volume, period=20):
    """Chaikin Money Flow — accumulation/distribution over N periods."""
    import numpy as np
    mfv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan) * volume
    return mfv.rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)


def donchian(high, low, period=20):
    """Donchian Channel. Returns (upper, middle, lower)."""
    upper = high.rolling(period).max()
    lower = low.rolling(period).min()
    middle = (upper + lower) / 2
    return upper, middle, lower


def pivot_points(high, low, close):
    """Daily pivot points. Returns (pivot, r1, r2, s1, s2)."""
    p = (high + low + close) / 3
    r1 = 2 * p - low
    r2 = p + (high - low)
    s1 = 2 * p - high
    s2 = p - (high - low)
    return p, r1, r2, s1, s2


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

INDICATORS: dict[str, dict[str, Any]] = {
    # Price Trend
    "sma":  {"fn": sma,  "params": {"period": "int"}, "desc": "Simple Moving Average", "category": "trend"},
    "ema":  {"fn": ema,  "params": {"period": "int"}, "desc": "Exponential Moving Average", "category": "trend"},
    "wma":  {"fn": wma,  "params": {"period": "int"}, "desc": "Weighted Moving Average", "category": "trend"},
    "hma":  {"fn": hma,  "params": {"period": "int"}, "desc": "Hull Moving Average", "category": "trend"},
    "macd":       {"fn": macd_line,  "params": {"fast": "int", "slow": "int"}, "desc": "MACD Line", "category": "trend"},
    "macd_signal":{"fn": macd_signal,"params": {"fast": "int", "slow": "int", "signal": "int"}, "desc": "MACD Signal", "category": "trend"},
    "macd_hist":  {"fn": macd_hist,  "params": {"fast": "int", "slow": "int", "signal": "int"}, "desc": "MACD Histogram", "category": "trend"},

    # Momentum
    "rsi":        {"fn": rsi,        "params": {"period": "int"}, "desc": "Relative Strength Index", "category": "momentum"},
    "stoch":      {"fn": stoch,      "params": {"k_period": "int", "d_period": "int"}, "desc": "Stochastic Oscillator", "category": "momentum"},
    "williams_r": {"fn": williams_r, "params": {"period": "int"}, "desc": "Williams %R", "category": "momentum"},
    "cci":        {"fn": cci,        "params": {"period": "int"}, "desc": "Commodity Channel Index", "category": "momentum"},
    "roc":        {"fn": roc,        "params": {"period": "int"}, "desc": "Rate of Change", "category": "momentum"},
    "momentum":   {"fn": momentum,   "params": {"period": "int"}, "desc": "Raw Momentum", "category": "momentum"},

    # Volatility
    "atr":      {"fn": atr,      "params": {"period": "int"}, "desc": "Average True Range", "category": "volatility"},
    "natr":     {"fn": natr,     "params": {"period": "int"}, "desc": "Normalized ATR (%)", "category": "volatility"},
    "bb_pct_b": {"fn": lambda close, **kw: bollinger_bands(close, kw.get("period", 20), kw.get("std", 2.0))[1],
                 "params": {"period": "int", "std": "float"}, "desc": "Bollinger %B", "category": "volatility"},
    "bb_width": {"fn": lambda close, **kw: (bollinger_bands(close, kw.get("period", 20), kw.get("std", 2.0))[1] -
                                             bollinger_bands(close, kw.get("period", 20), kw.get("std", 2.0))[2]) / close * 100,
                 "params": {"period": "int", "std": "float"}, "desc": "Bollinger Band Width (%)", "category": "volatility"},
    "keltner":  {"fn": keltner,  "params": {"period": "int", "atr_mult": "float"}, "desc": "Keltner Channel", "category": "volatility"},

    # Volume
    "volume":    {"fn": lambda close, **kw: None, "params": {}, "desc": "Raw Volume (from data, not computed)", "category": "volume"},
    "vol_sma":   {"fn": lambda volume, **kw: sma(volume, kw.get("period", 20)), "params": {"period": "int"}, "desc": "Volume Moving Average", "category": "volume"},
    "vol_ratio": {"fn": lambda volume, **kw: volume / sma(volume, kw.get("period", 20)).replace(0, float("nan")),
                  "params": {"period": "int"}, "desc": "Volume Ratio", "category": "volume"},
    "obv":       {"fn": obv,       "params": {}, "desc": "On-Balance Volume", "category": "volume"},
    "vwap":      {"fn": vwap,      "params": {"period": "int"}, "desc": "Volume-Weighted Average Price", "category": "volume"},
    "mfi":       {"fn": mfi,       "params": {"period": "int"}, "desc": "Money Flow Index", "category": "volume"},
    "vol_delta": {"fn": vol_delta, "params": {"period": "int"}, "desc": "Volume Delta (buying/selling pressure)", "category": "volume"},
    "cmf":       {"fn": cmf,       "params": {"period": "int"}, "desc": "Chaikin Money Flow", "category": "volume"},

    # Price Structure
    "hl_range":   {"fn": lambda high, low, close, **kw: (high - low) / close * 100,
                   "params": {}, "desc": "High-Low Range (%)", "category": "structure"},
    "price_loc":  {"fn": lambda high, low, close, **kw: (close - low.rolling(kw.get("period", 20)).min()) /
                   (high.rolling(kw.get("period", 20)).max() - low.rolling(kw.get("period", 20)).min()).replace(0, float("nan")),
                   "params": {"period": "int"}, "desc": "Price Location in Range", "category": "structure"},
    "donchian":   {"fn": donchian,  "params": {"period": "int"}, "desc": "Donchian Channel", "category": "structure"},
    "pivot":      {"fn": pivot_points, "params": {}, "desc": "Pivot Points", "category": "structure"},
}


def get_indicator(name: str) -> Callable:
    """Fetch an indicator function by name."""
    if name not in INDICATORS:
        available = "\n  ".join(sorted(INDICATORS.keys()))
        raise KeyError(f"Unknown indicator '{name}'. Available:\n  {available}")
    return INDICATORS[name]["fn"]


def list_indicators(category: str | None = None) -> str:
    """Return a formatted list of available indicators."""
    lines = []
    items = INDICATORS.items() if category is None else \
            [(k, v) for k, v in INDICATORS.items() if v.get("category") == category]

    for name, meta in sorted(items, key=lambda x: x[0]):
        params_str = ", ".join(f"{k}: {v}" for k, v in meta["params"].items())
        cat = f"[{meta['category']}]" if "category" in meta else ""
        lines.append(f"  {name:<15} {cat:<12} {meta['desc']}{'  (' + params_str + ')' if params_str else ''}")

    return "\n".join(lines)
