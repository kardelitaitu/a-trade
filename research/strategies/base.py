"""
Base Strategy abstract class for QuantumEdge.

All strategies must implement this interface.
See docs/strategy-guidelines.md for full specs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base for all trading strategies.

    Parameters
    ----------
    config : dict
        Flat dictionary of strategy parameters (no nested structures).
        Defaults to class-level DEFAULT_CONFIG if not provided.
    """

    # Class-level default config — override in subclasses
    DEFAULT_CONFIG: dict[str, Any] = {}

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        self._validate_config()

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from OHLCV data.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV DataFrame with columns [open, high, low, close, volume]
            and a datetime index (5-min frequency).

        Returns
        -------
        pd.Series
            Signals with same index as ``data``:
            -1 = short, 0 = neutral/no position, +1 = long.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name (e.g. 'MA Crossover')."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-paragraph description of the strategy logic."""

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def set_params(self, **kwargs) -> None:
        """Override one or more strategy parameters at runtime."""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
            else:
                raise KeyError(f"Unknown parameter: {key}")

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.config.items())
        return f"{self.name}({params})"

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Hook for subclasses to validate config on init. No-op by default."""
