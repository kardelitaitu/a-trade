"""
Strategy Factory for QuantumEdge.

Builds strategy instances from config dicts using the indicator registry.
Enables strategies to be defined by string names in config files or CLI.
"""

from __future__ import annotations

from typing import Any

from research.strategies.base import BaseStrategy
from research.features.registry import get_indicator


def create_strategy(
    strategy_type: str,
    config: dict[str, Any] | None = None,
) -> BaseStrategy:
    """
    Build a strategy instance from a type name and config dict.

    Parameters
    ----------
    strategy_type : str
        One of: 'ma_crossover', 'donchian_breakout', 'mean_reversion',
                'volatility_breakout'
    config : dict, optional
        Strategy parameters. Indicator names can be specified as strings
        (e.g. ``fast_indicator="ema"``) which are resolved via the registry.

    Returns
    -------
    BaseStrategy instance.
    """
    config = config or {}

    if strategy_type == "ma_crossover":
        from research.strategies.ma_crossover import MACrossover
        cfg = _resolve_indicator_params(config, ["fast_indicator", "slow_indicator"])
        return MACrossover(cfg)

    elif strategy_type == "donchian_breakout":
        from research.strategies.breakout import DonchianBreakout
        return DonchianBreakout(config)

    elif strategy_type == "mean_reversion":
        from research.strategies.mean_reversion import MeanReversion
        return MeanReversion(config)

    elif strategy_type == "volatility_breakout":
        from research.strategies.volatility_breakout import VolatilityBreakout
        return VolatilityBreakout(config)

    elif strategy_type == "volatility_squeeze":
        from research.strategies.volatility_squeeze import VolatilitySqueeze
        return VolatilitySqueeze(config)

    else:
        available = ["ma_crossover", "donchian_breakout", "mean_reversion", "volatility_breakout"]
        raise ValueError(f"Unknown strategy '{strategy_type}'. Available: {available}")


def _resolve_indicator_params(
    config: dict,
    indicator_keys: list[str],
) -> dict:
    """
    Resolve indicator name strings to validated params.
    E.g. ``fast_indicator="ema"`` stays as-is (the strategy uses it directly).
    Future: could validate that the name exists in the registry.
    """
    for key in indicator_keys:
        if key in config:
            name = config[key]
            # Verify it exists in registry
            try:
                get_indicator(name)
            except KeyError:
                raise ValueError(
                    f"Unknown indicator '{name}' for param '{key}'. "
                    f"Use list_indicators() to see available indicators."
                )
    return config


def list_strategies() -> str:
    """Return formatted list of available strategy types and their params."""
    lines = []
    strategies = {
        "ma_crossover": {
            "desc": "Moving Average Crossover",
            "params": {
                "fast_period": "int", "slow_period": "int",
                "fast_indicator": "str (sma/ema)", "slow_indicator": "str (sma/ema)",
                "ma_type": "str (sma/ema) [deprecated]",
                "filter_volume_pct": "float (optional)",
                "filter_atr_mult": "float (optional)",
            },
        },
        "donchian_breakout": {
            "desc": "Donchian Channel Breakout",
            "params": {
                "entry_period": "int", "exit_period": "int",
                "filter_volume_pct": "float (optional)",
                "filter_atr_mult": "float (optional)",
            },
        },
        "mean_reversion": {
            "desc": "Mean Reversion (RSI or Bollinger)",
            "params": {
                "mode": "str (rsi/bollinger)",
                "rsi_period": "int", "rsi_oversold": "float", "rsi_overbought": "float",
                "bb_period": "int", "bb_std": "float",
                "filter_volume_pct": "float (optional)",
            },
        },
        "volatility_breakout": {
            "desc": "Volatility Breakout (ATR expansion)",
            "params": {
                "atr_period": "int", "atr_multiplier": "float",
                "atr_lookback": "int", "min_hold": "int",
                "filter_volume_pct": "float (optional)",
            },
        },
        "volatility_squeeze": {
            "desc": "Volatility Squeeze (BB/Keltner squeeze)",
            "params": {
                "bb_period": "int", "bb_std": "float",
                "keltner_period": "int", "keltner_atr_mult": "float",
                "squeeze_lookback": "int", "min_hold": "int",
                "filter_volume_pct": "float (optional)",
            },
        },
    }

    for name, meta in sorted(strategies.items()):
        lines.append(f"\n{name:<25} {meta['desc']}")
        lines.append("  " + "-" * 60)
        for pname, ptype in meta["params"].items():
            lines.append(f"  {pname:<30} {ptype}")
    return "\n".join(lines)
