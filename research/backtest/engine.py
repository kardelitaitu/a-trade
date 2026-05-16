"""
Vectorized backtesting engine for QuantumEdge.

Processes OHLCV data + signal series to produce equity curves
and trade logs using pure vectorized operations (no loops over candles).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "initial_capital": 10_000.0,  # USDT
    "fee": 0.00075,               # 0.075% per trade (maker+taker average)
    "slippage": 0.0001,           # 0.01% per trade (1 tick on BTC)
    "position_size_pct": 1.0,     # 100% of capital per signal
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
            Values: -1 (short), 0 (neutral), +1 (long).

        Returns
        -------
        BacktestResult
        """
        self._result = None

        # Align signals to data
        signals = signals.reindex(self.data.index, fill_value=0).astype(float)

        close = self.data["close"]
        capital = self._config["initial_capital"]
        fee_rate = self._config["fee"]
        slip_rate = self._config["slippage"]
        total_cost_rate = fee_rate + slip_rate

        # ---------- Position (shift by 1 to avoid look-ahead) ----------
        positions = signals.shift(1).fillna(0)
        position_changes = positions.diff().fillna(0)

        # ---------- Price returns (gross) ----------
        # close_return[t] = close[t] / close[t-1] - 1
        close_return = close.pct_change().fillna(0)

        # Strategy return = position * price return
        strategy_return = positions * close_return

        # ---------- Transaction costs ----------
        # Cost per trade = abs(position_change) * cost_rate (no price multiplication,
        # since we express cost as fraction of capital)
        change_magnitude = position_changes.abs()

        # Costs when position changes: entry + exit. Each change costs fee + slippage.
        # 1 unit of position costs cost_rate to establish, another cost_rate to close.
        # But position_changes already captures the delta. To avoid double-counting,
        # cost = abs(delta) * cost_rate
        cost_rate = change_magnitude * total_cost_rate

        # Net return
        net_return = strategy_return - cost_rate

        # ---------- Equity curve ----------
        equity = (1 + net_return).cumprod() * capital

        # ---------- Trade log ----------
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
    # Trade log extraction
    # ------------------------------------------------------------------

    def _extract_trades(
        self,
        signals: pd.Series,
        positions: pd.Series,
        close: pd.Series,
        equity: pd.Series,
    ) -> list[Trade]:
        """Build trade list from position changes."""
        trades: list[Trade] = []
        capital = self._config["initial_capital"]
        fee_rate = self._config["fee"] + self._config["slippage"]

        entry_idx: Optional[int] = None
        entry_side: int = 0

        prev_position = 0.0
        for i in range(len(positions)):
            curr_position = positions.iloc[i]

            if curr_position == prev_position:
                continue

            # Position changed — determine what happened
            went_to_zero = curr_position == 0
            came_from_zero = prev_position == 0

            if came_from_zero and not went_to_zero:
                # Opening a NEW position (0 → 1 or 0 → -1)
                entry_idx = i
                entry_side = int(curr_position)

            elif went_to_zero and not came_from_zero:
                # Closing an existing position (1 → 0 or -1 → 0)
                if entry_idx is not None:
                    side = entry_side
                    entry_time = positions.index[entry_idx]
                    exit_time = positions.index[i]
                    entry_price = float(close.iloc[entry_idx])
                    exit_price = float(close.iloc[i])
                    cap_at_entry = equity.iloc[entry_idx - 1] if entry_idx > 0 else capital
                    size_units = (cap_at_entry * self._config["position_size_pct"]) / entry_price
                    price_move = (exit_price / entry_price - 1)
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
                entry_idx = None
                entry_side = 0

            else:
                # Position FLIP (1 → -1 or -1 → 1)
                if entry_idx is not None:
                    # Close the current position first
                    side = entry_side
                    entry_time = positions.index[entry_idx]
                    exit_time = positions.index[i]
                    entry_price = float(close.iloc[entry_idx])
                    exit_price = float(close.iloc[i])
                    cap_at_entry = equity.iloc[entry_idx - 1] if entry_idx > 0 else capital
                    size_units = (cap_at_entry * self._config["position_size_pct"]) / entry_price
                    price_move = (exit_price / entry_price - 1)
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
                # Then open the new flipped position
                entry_idx = i
                entry_side = int(curr_position)

            prev_position = curr_position

        return trades
