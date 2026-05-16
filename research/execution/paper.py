"""
Paper trading pipeline for QuantumEdge.

Generates live trading signals from current market data and logs them
to results/paper/ for tracking. Can run as a scheduled cron job.

Each strategy is tracked independently with its own capital allocation,
position, and PnL. Fees match the VectorizedBacktest engine's total cost
rate (fee + slippage).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from research.backtest.engine import VectorizedBacktest, Trade
from research.backtest.metrics import compute_metrics, format_metrics_report

logger = logging.getLogger(__name__)

# Total cost rate matching VectorizedBacktest engine: fee 0.0001 + slippage 0.00001
DEFAULT_FEE_RATE = 0.00011

# Best parameters discovered during optimization
OPTIMIZED_STRATEGIES = {
    "ma_crossover": {
        "module": "research.strategies.ma_crossover",
        "class": "MACrossover",
        "params": {"fast_period": 12, "slow_period": 26, "ma_type": "ema"},
    },
    "donchian_breakout": {
        "module": "research.strategies.breakout",
        "class": "DonchianBreakout",
        "params": {"entry_period": 30, "exit_period": 15},
    },
    "mean_reversion_rsi": {
        "module": "research.strategies.mean_reversion",
        "class": "MeanReversion",
        "params": {
            "mode": "rsi", "rsi_period": 25,
            "rsi_oversold": 30, "rsi_overbought": 88,
        },
    },
}


def _import_strategy(config: dict):
    """Import a strategy class by module path."""
    import importlib
    mod = importlib.import_module(config["module"])
    cls = getattr(mod, config["class"])
    return cls(config["params"])


@dataclass
class PaperTradeSignal:
    """A single paper trading signal event."""
    timestamp: str
    strategy: str
    signal: float           # -1 to +1
    close_price: float
    position: float         # Current position after this signal
    portfolio_value: float  # Per-strategy capital after this signal


class PaperTrader:
    """
    Paper trading engine. Generates and logs signals without real execution.

    Each strategy runs independently with its own capital allocation,
    position tracking, PnL computation, and fee deductions.

    Parameters
    ----------
    data_dir : Path
        Directory with OHLCV parquet files.
    results_dir : Path
        Directory for paper trading logs.
    initial_capital : float
        Total initial capital split evenly across strategies.
    fee_rate : float
        Total cost rate (fee + slippage). Defaults to engine's 0.00011.
    strategies : dict, optional
        Strategy configurations. Defaults to OPTIMIZED_STRATEGIES.
    """

    def __init__(
        self,
        data_dir: Path = Path("data/processed"),
        results_dir: Path = Path("results/paper"),
        initial_capital: float = 10_000.0,
        fee_rate: float = DEFAULT_FEE_RATE,
        strategies: Optional[dict] = None,
    ):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.strategies = strategies or OPTIMIZED_STRATEGIES

        # Per-strategy tracked state — each strategy is independent
        self._strategy_state: dict[str, dict] = {}  # name -> {capital, position, last_price}
        self._signal_log: list[PaperTradeSignal] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_data(self, symbol: str = "btc_usdt", interval: str = "5m") -> pd.DataFrame:
        """Load latest OHLCV data for paper trading."""
        path = self.data_dir / f"{symbol}_{interval}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No data at {path}")
        return pd.read_parquet(path)

    def run(self, symbol: str = "btc_usdt") -> list[PaperTradeSignal]:
        """
        Run all strategies on the latest data and generate signals.

        Each strategy's state is tracked independently:
        - Capital is allocated evenly across strategies on first run
        - PnL is computed from price movement on existing position
        - Fees are deducted on position changes
        - Only new signals are appended to the log file

        Returns list of PaperTradeSignal (one per strategy).
        """
        data = self.load_data(symbol)
        last_price = float(data["close"].iloc[-1])
        new_signals = []
        n_strategies = len(self.strategies)

        for name, config in self.strategies.items():
            try:
                strat = _import_strategy(config)
                sig = strat.generate_signals(data)
                signal_value = float(sig.iloc[-1])

                # Initialize per-strategy state on first run
                if name not in self._strategy_state:
                    self._strategy_state[name] = {
                        "capital": self.initial_capital / n_strategies,
                        "position": 0.0,
                        "last_price": last_price,
                    }

                state = self._strategy_state[name]
                old_position = state["position"]
                old_price = state["last_price"]
                capital = state["capital"]

                # --- PnL from existing position's price movement ---
                # If we hold a position, the change in price affects our capital
                pnl = 0.0
                if old_position != 0.0 and old_price > 0:
                    price_return = (last_price / old_price) - 1.0
                    pnl = old_position * capital * price_return

                # --- Fee on position change ---
                # Charged on the absolute change in position size
                pos_change = signal_value - old_position
                fee = abs(pos_change) * capital * self.fee_rate

                # --- Update capital and position ---
                new_capital = capital + pnl - fee
                self._strategy_state[name] = {
                    "capital": new_capital,
                    "position": signal_value,
                    "last_price": last_price,
                }

                record = PaperTradeSignal(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    strategy=name,
                    signal=signal_value,
                    close_price=last_price,
                    position=signal_value,
                    portfolio_value=new_capital,
                )
                new_signals.append(record)
                logger.info(f"{name}: signal={signal_value:.4f} @ {last_price:.2f}, "
                            f"capital={new_capital:.2f}, pnl={pnl:+.2f}, fee={fee:.4f}")

            except Exception as e:
                logger.warning(f"{name}: failed — {e}")

        # Append only new signals to the persistent log
        self._signal_log.extend(new_signals)
        self._save_log(new_signals)
        return new_signals

    def summary(self) -> str:
        """
        Compute performance metrics from all tracked signals.

        Builds a total portfolio equity curve by summing per-strategy
        capital at each signal event, then computes standard metrics.
        """
        if not self._signal_log:
            return "No signals yet."

        # Build dataframe from all signal events
        df = pd.DataFrame([asdict(s) for s in self._signal_log])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        # Total portfolio equity curve — sum per-strategy capital at each timestamp
        if "portfolio_value" in df.columns:
            equity = df.groupby("timestamp")["portfolio_value"].sum()
        else:
            equity = pd.Series([], dtype=float)

        # Reconstruct trades from per-strategy consecutive signals
        trades = self._reconstruct_trades(df)

        metrics = compute_metrics(equity, trades)
        return format_metrics_report(metrics)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_log(self, new_signals: list[PaperTradeSignal]) -> None:
        """
        Append only new signals to the paper trading log file.

        Fixes the duplicate-log bug: previously all historical signals
        were re-written on every run. Now only fresh signals are appended.
        """
        if not new_signals:
            return
        path = self.results_dir / "paper_trades.jsonl"
        with open(path, "a") as f:
            for s in new_signals:
                f.write(json.dumps(asdict(s)) + "\n")
        logger.info(f"Log saved ({len(new_signals)} new entries)")

    def _reconstruct_trades(self, df: pd.DataFrame) -> list[Trade]:
        """
        Reconstruct Trade objects from signal history for metrics reporting.

        Each strategy's signal series is scanned for direction changes
        (non-zero → zero, zero → non-zero, flip). Each segment yields a
        single trade with computed PnL and fees.
        """
        trades: list[Trade] = []

        for name in self.strategies:
            strat = df[df["strategy"] == name].sort_values("timestamp")
            if len(strat) < 2:
                continue

            positions = strat["position"].values
            prices = strat["close_price"].values
            times = strat["timestamp"].values

            entry_idx = -1
            entry_side = 0

            for i in range(len(positions)):
                curr = positions[i]
                prev = positions[i - 1] if i > 0 else 0.0

                # Determine direction (same logic as extract_trades_numba)
                curr_dir = 0
                if curr > 0:
                    curr_dir = 1
                elif curr < 0:
                    curr_dir = -1

                prev_dir = 0
                if prev > 0:
                    prev_dir = 1
                elif prev < 0:
                    prev_dir = -1

                if curr_dir == prev_dir:
                    continue  # Same direction, no trade boundary

                if curr_dir == 0:
                    # Position went to zero — close existing trade
                    if entry_idx >= 0:
                        trades.append(self._build_trade(
                            strat, entry_idx, i, entry_side
                        ))
                        entry_idx = -1
                        entry_side = 0
                elif prev_dir == 0:
                    # Position opened from zero
                    entry_idx = i
                    entry_side = curr_dir
                else:
                    # Direction flip — close old, open new at same index
                    if entry_idx >= 0:
                        trades.append(self._build_trade(
                            strat, entry_idx, i, entry_side
                        ))
                    entry_idx = i
                    entry_side = curr_dir

        return trades

    def _build_trade(
        self,
        strat: pd.DataFrame,
        entry_idx: int,
        exit_idx: int,
        side: int,
    ) -> Trade:
        """Build a Trade from a segment of a strategy's signal series."""
        entry_time = pd.Timestamp(strat.iloc[entry_idx]["timestamp"])
        exit_time = pd.Timestamp(strat.iloc[exit_idx]["timestamp"])
        entry_price = float(strat.iloc[entry_idx]["close_price"])
        exit_price = float(strat.iloc[exit_idx]["close_price"])
        position_size = abs(float(strat.iloc[entry_idx]["position"]))
        entry_cap = float(strat.iloc[entry_idx]["portfolio_value"])

        # Gross return
        price_move = (exit_price / entry_price) - 1.0
        gross_return = -price_move if side == -1 else price_move

        # PnL and fees
        pnl = entry_cap * position_size * gross_return
        # Approx fees: entry + exit at position size
        fees = 2 * position_size * entry_cap * self.fee_rate

        return Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            size=position_size,
            pnl=pnl,
            pnl_pct=(pnl / entry_cap * 100) if entry_cap > 0 else 0.0,
            return_pct=gross_return * 100,
            fees=fees,
            duration=str(exit_time - entry_time),
        )

    @property
    def total_portfolio_value(self) -> float:
        """Sum of all per-strategy capital."""
        if not self._strategy_state:
            return self.initial_capital
        return sum(s["capital"] for s in self._strategy_state.values())




def paper_trade_job(symbol: str = "btc_usdt") -> str:
    """
    Entry point for cron job: run paper trading and return summary.
    """
    trader = PaperTrader()
    signals = trader.run(symbol)
    if not signals:
        return "No signals generated."

    summary = trader.summary()
    total_value = trader.total_portfolio_value
    return (
        f"Paper trade complete. {len(signals)} signals generated.\n"
        f"Portfolio value: ${total_value:,.2f}\n{summary}"
    )
