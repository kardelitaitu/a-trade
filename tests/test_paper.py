"""Tests for research/execution/paper.py."""

from research.execution.paper import PaperTrader

class TestPaperTrader:

    def test_import(self):
        """Should import without error."""
        from research.execution import paper
        assert hasattr(paper, "PaperTrader")
        assert hasattr(paper, "PaperTradeSignal")
