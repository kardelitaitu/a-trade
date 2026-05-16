"""
Vectorized backtesting engine for QuantumEdge.

Processes OHLCV data + signal series to produce equity curves
and trade logs using pure vectorized operations.
Trade extraction is numba-accelerated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from research.backtest._numba_ops import extract_trades_numba

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "initial_capital": 10_000.0,
    "fee": 0.00075,
    "slippage": 0.0001,
    "position_size_pct": 1.0,
    "size_mode": "fixed",
}


@dataclass
class Trade:
    """A single closed trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: int
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    return_pct: float
    fees: float
    duration: str


@dataclass
class BacktestResult:
    """Container for backtest outputs."""
    equity_curve: pd.Series
    trades: list[Trade]
    signals: pd.Series
    positions: pd.Series
    config: dict = field(default_factory=lambda: DEFAULT_CONFIG.copy())


class VectorizedBacktest:
    """
    Vectorized backtesting engine.

    Parameters
    ----------
    data : pd.DataFrame
        OHLCV data with datetime index. Must have columns:
        open, high, low, close, volume.
    config : dict, optional
        Override DEFAULT_CONFIG settings.
    """

    def __init__(self, data: pd.DataFrame, config: Optional[dict] = None):
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self.data = data
        self._config = DEFAULT_CONFIG.copy()
        if config:
            self._config.update(config)
        self._result: Optional[BacktestResult] = None

    @property
    def config(self) -> dict:
        return self._config.copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, signals: pd.Series) -> BacktestResult:
        """
        Execute a vectorized backtest.

        Parameters
        ----------
        signals : pd.Series
            Trading signals aligned with ``self.data.index``.
            Values: -1, 0, +1 (or continuous float).

        Returns
        -------
        BacktestResult
        """
        self._result = None

        signals = signals.reindex(self.data.index, fill_value=0).astype(float)
        close = self.data["close"]
        capital = self._config["initial_capital"]
        total_cost_rate = self._config["fee"] + self._config["slippage"]

        # Position (shift by 1 to avoid look-ahead)
        positions = signals.shift(1).fillna(0)
        position_changes = positions.diff().fillna(0)

        # Strategy returns
        close_return = close.pct_change(fill_method=None).fillna(0)
        strategy_return = positions * close_return

        # Transaction costs
        cost_rate = position_changes.abs() * total_cost_rate
        net_return = strategy_return - cost_rate

        # Equity curve
        equity = (1 + net_return).cumprod() * capital

        # Trade log (numba-accelerated)
        trades = self._extract_trades(signals, positions, close, equity)

        self._result = BacktestResult(
            equity_curve=equity,
            trades=trades,
            signals=signals,
            positions=positions,
            config=self._config,
        )
        return self._result

    # ------------------------------------------------------------------
    # Trade log extraction (numba accelerated)
    # ------------------------------------------------------------------

    def _extract_trades(
        self,
        signals: pd.Series,
        positions: pd.Series,
        close: pd.Series,
        equity: pd.Series,
    ) -> list[Trade]:
        """Build trade list from position changes using numba."""
        pos_arr = positions.values.astype(np.float64)
        close_arr = close.values.astype(np.float64)
        equity_arr = equity.values.astype(np.float64)

        capital = self._config["initial_capital"]
        fee_rate = self._config["fee"] + self._config["slippage"]
        size_pct = self._config["position_size_pct"]

        entry_idxs, exit_idxs, sides = extract_trades_numba(pos_arr)

        trades: list[Trade] = []
        for k in range(len(entry_idxs)):
            ei = entry_idxs[k]
            xi = exit_idxs[k]
            side = int(sides[k])

            entry_time = positions.index[ei]
            exit_time = positions.index[xi]
            entry_price = float(close_arr[ei])
            exit_price = float(close_arr[xi])

            cap_at_entry = equity_arr[ei - 1] if ei > 0 else capital
            size_units = (cap_at_entry * size_pct) / entry_price

            price_move = (exit_price / entry_price) - 1
            gross_return = -price_move if side == -1 else price_move

            fees = entry_price * size_units * fee_rate + exit_price * size_units * fee_rate
            pnl = (exit_price - entry_price) * size_units * side - fees
            pnl_pct = (pnl / cap_at_entry) * 100
            duration = str(exit_time - entry_time)

            trades.append(Trade(
                entry_time=entry_time, exit_time=exit_time, side=side,
                entry_price=entry_price, exit_price=exit_price,
                size=size_units, pnl=pnl, pnl_pct=pnl_pct,
                return_pct=gross_return * 100, fees=fees, duration=duration,
            ))

        return trades
