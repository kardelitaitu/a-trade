"""Tests for research/execution/paper.py."""

from pathlib import Path

import pandas as pd
import pytest

from research.execution.paper import (
    PaperTrader,
    PaperTradeSignal,
    DEFAULT_FEE_RATE,
    _import_strategy,
)


class TestPaperTradeSignal:
    """PaperTradeSignal dataclass tests."""

    def test_create_signal(self):
        s = PaperTradeSignal(
            timestamp="2024-01-01T00:00:00",
            strategy="ma_crossover",
            signal=0.5,
            close_price=50000.0,
            position=0.5,
            portfolio_value=10000.0,
        )
        assert s.strategy == "ma_crossover"
        assert s.signal == 0.5


class TestPaperTrader:
    """PaperTrader tests."""

    def test_default_fee_rate_matches_engine(self):
        """DEFAULT_FEE_RATE should match engine's total cost rate (fee + slippage)."""
        from research.backtest.engine import DEFAULT_CONFIG
        engine_total = DEFAULT_CONFIG["fee"] + DEFAULT_CONFIG["slippage"]
        assert DEFAULT_FEE_RATE == pytest.approx(engine_total), (
            f"Paper trader fee_rate {DEFAULT_FEE_RATE} doesn't match engine total {engine_total}"
        )

    def test_initial_state(self, tmp_path):
        """Trader initializes with no state and correct defaults."""
        trader = PaperTrader(
            data_dir=tmp_path,
            results_dir=tmp_path / "results",
        )
        assert trader.initial_capital == 10_000.0
        assert trader.fee_rate == DEFAULT_FEE_RATE
        assert trader._signal_log == []
        assert trader._strategy_state == {}
        assert trader.total_portfolio_value == trader.initial_capital

    def test_custom_fee_rate(self, tmp_path):
        """Custom fee_rate should be used instead of default."""
        trader = PaperTrader(
            data_dir=tmp_path,
            results_dir=tmp_path / "results",
            fee_rate=0.001,
        )
        assert trader.fee_rate == 0.001

    def test_no_signals_summary(self):
        """summary() should return a message when no signals exist."""
        trader = PaperTrader(data_dir=Path("/nonexistent"), results_dir=Path("/nonexistent"))
        result = trader.summary()
        assert "No signals yet" in result

    def test_strategy_state_isolation(self, tmp_path):
        """Per-strategy state should be isolated — modifying one doesn't affect others."""
        trader = PaperTrader(
            data_dir=tmp_path,
            results_dir=tmp_path / "results",
        )
        # Manually seed independent states
        trader._strategy_state["strat_a"] = {"capital": 5000.0, "position": 0.5, "last_price": 100.0}
        trader._strategy_state["strat_b"] = {"capital": 5000.0, "position": -0.3, "last_price": 100.0}

        assert trader._strategy_state["strat_a"]["position"] == 0.5
        assert trader._strategy_state["strat_b"]["position"] == -0.3
        assert trader._strategy_state["strat_a"]["capital"] == 5000.0

        # Modify one
        trader._strategy_state["strat_a"]["capital"] = 5500.0
        assert trader._strategy_state["strat_b"]["capital"] == 5000.0  # Unchanged

    def test_total_portfolio_value_sum(self, tmp_path):
        """total_portfolio_value should sum all per-strategy capital."""
        trader = PaperTrader(
            data_dir=tmp_path,
            results_dir=tmp_path / "results",
        )
        trader._strategy_state["a"] = {"capital": 3000.0, "position": 0.0, "last_price": 100.0}
        trader._strategy_state["b"] = {"capital": 7000.0, "position": 0.0, "last_price": 100.0}
        assert trader.total_portfolio_value == 10_000.0

    def test_save_log_only_new(self, tmp_path):
        """_save_log should only write new signals, not all historical ones."""
        trader = PaperTrader(
            data_dir=tmp_path,
            results_dir=tmp_path / "results",
        )

        # Initial batch
        s1 = PaperTradeSignal("t1", "strat_a", 0.5, 100.0, 0.5, 5000.0)
        trader._save_log([s1])

        log_path = tmp_path / "results" / "paper_trades.jsonl"
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1

        # Second batch — only new signals should be appended
        s2 = PaperTradeSignal("t2", "strat_b", -0.3, 101.0, -0.3, 5000.0)
        trader._save_log([s2])

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2  # Still only 2 total, not 4

    def test_import_strategy(self):
        """_import_strategy should instantiate a strategy from config dict."""
        config = {
            "module": "research.strategies.ma_crossover",
            "class": "MACrossover",
            "params": {"fast_period": 10, "slow_period": 30},
        }
        strat = _import_strategy(config)
        assert strat.name == "MA Crossover"
        assert strat.config["fast_period"] == 10


class TestPaperTradeJob:
    """paper_trade_job integration tests."""

    def test_job_exists(self):
        """paper_trade_job function should be importable."""
        from research.execution.paper import paper_trade_job
        assert callable(paper_trade_job)
