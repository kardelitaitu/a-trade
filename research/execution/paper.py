"""
Paper trading pipeline for QuantumEdge.

Generates live trading signals from current market data and logs them
to results/paper/ for tracking. Can run as a scheduled cron job.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from research.backtest.engine import VectorizedBacktest
from research.backtest.metrics import compute_metrics

logger = logging.getLogger(__name__)

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
    portfolio_value: float


class PaperTrader:
    """
    Paper trading engine. Generates and logs signals without real execution.

    Parameters
    ----------
    data_dir : Path
        Directory with OHLCV parquet files.
    results_dir : Path
        Directory for paper trading logs.
    initial_capital : float
    strategies : dict, optional
        Strategy configurations. Defaults to OPTIMIZED_STRATEGIES.
    """

    def __init__(
        self,
        data_dir: Path = Path("data/processed"),
        results_dir: Path = Path("results/paper"),
        initial_capital: float = 10_000.0,
        strategies: Optional[dict] = None,
    ):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.initial_capital = initial_capital
        self.strategies = strategies or OPTIMIZED_STRATEGIES

        self._strategies: dict = {}
        self._capital = initial_capital
        self._position = 0.0
        self._signals: list[PaperTradeSignal] = []

    def load_data(self, symbol: str = "btc_usdt", interval: str = "5m") -> pd.DataFrame:
        """Load latest OHLCV data for paper trading."""
        path = self.data_dir / f"{symbol}_{interval}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No data at {path}")
        return pd.read_parquet(path)

    def run(self, symbol: str = "btc_usdt") -> list[PaperTradeSignal]:
        """
        Run all strategies on the latest data and generate signals.

        Returns list of PaperTradeSignal for each strategy.
        """
        data = self.load_data(symbol)
        signals = []

        for name, config in self.strategies.items():
            try:
                strat = _import_strategy(config)
                sig = strat.generate_signals(data)
                last_signal = float(sig.iloc[-1])
                last_price = float(data["close"].iloc[-1])

                # Update portfolio
                old_position = self._position
                self._position = last_signal
                trade_value = (self._position - old_position) * self._capital
                self._capital += trade_value

                record = PaperTradeSignal(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    strategy=name,
                    signal=last_signal,
                    close_price=last_price,
                    position=self._position,
                    portfolio_value=self._capital,
                )
                signals.append(record)
                logger.info(f"{name}: signal={last_signal:.2f} @ {last_price:.2f}")
            except Exception as e:
                logger.warning(f"{name}: failed — {e}")

        self._signals.extend(signals)
        self._save_log()
        return signals

    def _save_log(self) -> None:
        """Append signals to the paper trading log."""
        path = self.results_dir / "paper_trades.jsonl"
        with open(path, "a") as f:
            for s in self._signals:
                f.write(json.dumps(asdict(s)) + "\n")
        logger.info(f"Log saved ({len(self._signals)} entries)")

    def summary(self) -> str:
        """Print current paper trading summary."""
        from research.backtest.engine import VectorizedBacktest, Trade
        from research.backtest.metrics import format_metrics_report

        if not self._signals:
            return "No signals yet."

        # Build simple backtest from paper trades
        trades = []
        idx = pd.date_range("now", periods=1, freq="5min")

        for name in self.strategies:
            strategy_signals = [s for s in self._signals if s.strategy == name]
            if len(strategy_signals) < 2:
                continue
            for i in range(1, len(strategy_signals)):
                entry = strategy_signals[i - 1]
                exit = strategy_signals[i]
                trades.append(Trade(
                    entry_time=pd.Timestamp(entry.timestamp),
                    exit_time=pd.Timestamp(exit.timestamp),
                    side=1 if entry.signal > 0 else -1,
                    entry_price=entry.close_price,
                    exit_price=exit.close_price,
                    size=1.0, pnl=0, pnl_pct=0,
                    return_pct=0, fees=0, duration="",
                ))

        equity = pd.Series([s.portfolio_value for s in self._signals], index=range(len(self._signals)))
        metrics = compute_metrics(equity, trades)

        return format_metrics_report(metrics)


def paper_trade_job(symbol: str = "btc_usdt") -> str:
    """
    Entry point for cron job: run paper trading and return summary.
    """
    trader = PaperTrader()
    signals = trader.run(symbol)
    if not signals:
        return "No signals generated."

    summary = trader.summary()
    return f"Paper trade complete. {len(signals)} signals generated.\n{summary}"
