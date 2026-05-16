"""More bulk tests — monte carlo, parallel mc, sensitivity, ensemble, regime, proxies."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture
def c():
    np.random.seed(42)
    return np.array(100 + np.cumsum(np.random.randn(500)), dtype=float)


class TestMC:
    def test_ma_run(self, c):
        from research.optimization.monte_carlo import monte_carlo_ma
        r = monte_carlo_ma(c, (5, 15), (20, 40))
        assert r.n_combos > 0

    def test_ma_metrics_shape(self, c):
        from research.optimization.monte_carlo import monte_carlo_ma
        r = monte_carlo_ma(c, (5, 15), (20, 40))
        assert r.metrics.shape[1] == 6

    def test_ma_sorted(self, c):
        from research.optimization.monte_carlo import monte_carlo_ma
        r = monte_carlo_ma(c, (5, 15), (20, 40))
        assert r.sorted_indices[0] == r.best_idx

    def test_rsi_run(self, c):
        from research.optimization.monte_carlo import monte_carlo_rsi
        r = monte_carlo_rsi(c, [7,14], [25,30], [70,75])
        assert r.n_combos > 0

    def test_save_report(self, c, tmp_path):
        from research.optimization.monte_carlo import monte_carlo_ma, save_mc_report
        r = monte_carlo_ma(c, (5, 15), (20, 40))
        p = save_mc_report(r, output_dir=Path(tmp_path))
        assert Path(p).exists()

    def test_report_has_top(self, c, tmp_path):
        from research.optimization.monte_carlo import monte_carlo_ma, save_mc_report
        r = monte_carlo_ma(c, (5, 15), (20, 40))
        p = save_mc_report(r, output_dir=Path(tmp_path))
        assert "TOP" in open(p).read()

    def test_report_has_sensitivity(self, c, tmp_path):
        from research.optimization.monte_carlo import monte_carlo_ma, save_mc_report
        r = monte_carlo_ma(c, (5, 15), (20, 40))
        p = save_mc_report(r, output_dir=Path(tmp_path))
        assert "Sensitivity" in open(p).read() or "How to read" in open(p).read()

    def test_sltp_shape(self, c):
        from research.optimization.monte_carlo import monte_carlo_ma_sltp
        h = c+5; l = c-5
        r = monte_carlo_ma_sltp(c, h, l, (5,10), (20,30), [0.01], [0.03])
        assert r.n_combos > 0

    def test_best_metrics(self, c):
        from research.optimization.monte_carlo import monte_carlo_ma
        r = monte_carlo_ma(c, (5, 15), (20, 40))
        b = r.best_metrics()
        assert b.sharpe is not None and b.final_equity > 0

    def test_summary_string(self, c):
        from research.optimization.monte_carlo import monte_carlo_ma
        r = monte_carlo_ma(c, (5, 15), (20, 40))
        assert len(r.summary()) > 20


class TestParallelMC:
    @pytest.fixture
    def d(self):
        np.random.seed(42)
        idx = pd.date_range("2024-01-01", periods=200, freq="5min")
        cl = 100 + np.cumsum(np.random.randn(200) * 0.3)
        return pd.DataFrame({"open": cl-0.2, "high": cl+0.5, "low": cl-0.5, "close": cl, "volume": np.random.uniform(10, 100, 200)}, index=idx)

    def test_single(self, d):
        from research.optimization.parallel_mc import parallel_monte_carlo
        r = parallel_monte_carlo("ma_crossover", [{"fast_period":10,"slow_period":30}], d, n_jobs=1)
        assert len(r) == 1

    def test_multiple(self, d):
        from research.optimization.parallel_mc import parallel_monte_carlo
        r = parallel_monte_carlo("ma_crossover", [{"fast_period":i,"slow_period":i+20} for i in [5,10,15]], d, n_jobs=1)
        assert len(r) == 3

    def test_empty(self, d):
        from research.optimization.parallel_mc import parallel_monte_carlo
        assert parallel_monte_carlo("ma_crossover", [], d, n_jobs=1) == []

    def test_results_keys(self, d):
        from research.optimization.parallel_mc import parallel_monte_carlo
        r = parallel_monte_carlo("ma_crossover", [{"fast_period":10,"slow_period":30}], d, n_jobs=1)
        for k in ["sharpe","max_dd","trades","params"]:
            assert k in r[0]

    def test_format_report(self, d):
        from research.optimization.parallel_mc import parallel_monte_carlo, format_parallel_results
        r = parallel_monte_carlo("ma_crossover", [{"fast_period":10,"slow_period":30}], d, n_jobs=1)
        rep = format_parallel_results(r, strategy_name="test")
        assert "MONTE CARLO" in rep and "test" in rep

    def test_save_report(self, d, tmp_path):
        from research.optimization.parallel_mc import parallel_monte_carlo, save_parallel_results
        r = parallel_monte_carlo("ma_crossover", [{"fast_period":10,"slow_period":30}], d, n_jobs=1)
        p = save_parallel_results(r, "test", output_dir=str(tmp_path))
        assert Path(p).exists()


class TestSens:
    def make(self, n=8):
        return [{"params":{"a":i%4,"b":i%2},"sharpe":0.5-i*0.1,"max_dd":-10-i*2,
                 "profit_factor":1.2-i*0.1,"total_return":20-i*3,"trades":50-i*5} for i in range(n)]

    def test_analyze(self):
        from research.optimization.sensitivity import analyze_sensitivity
        a = analyze_sensitivity(self.make())
        assert "sharpe" in a and "drawdown" in a and "calmar" in a

    def test_recs_nonempty(self):
        from research.optimization.sensitivity import analyze_sensitivity
        a = analyze_sensitivity(self.make())
        assert len(a["recommendations"]) > 100

    def test_empty(self):
        from research.optimization.sensitivity import analyze_sensitivity
        a = analyze_sensitivity([])
        assert "No results" in a["recommendations"]

    def test_save(self, tmp_path):
        from research.optimization.sensitivity import analyze_sensitivity, save_sensitivity_report
        a = analyze_sensitivity(self.make())
        p = save_sensitivity_report(a, "t", output_dir=str(tmp_path))
        assert Path(p).exists()

    def test_best_combo(self):
        from research.optimization.sensitivity import analyze_sensitivity
        a = analyze_sensitivity(self.make())
        assert "BEST COMBO" in a["recommendations"]

    def test_how_to_read(self):
        from research.optimization.sensitivity import analyze_sensitivity
        a = analyze_sensitivity(self.make())
        assert "How to read" in a["recommendations"]


class TestEns:
    @pytest.fixture
    def ix(self):
        return pd.date_range("2024-01-01", periods=100, freq="5min")
    def test_rolling(self, ix):
        from research.strategies.ensemble import compute_rolling_sharpe
        c = pd.Series(100 + np.cumsum(np.random.randn(100)*0.3), index=ix)
        s = pd.Series(np.random.choice([-1,0,1], 100), index=ix)
        assert compute_rolling_sharpe(s, c, 30).notna().sum() > 0
    def test_equal_two(self, ix):
        from research.strategies.ensemble import ensemble_signals
        c = pd.Series(100 + np.arange(10), index=ix[:10])
        r = ensemble_signals({"a":pd.Series([1]*10,index=ix[:10]), "b":pd.Series([0]*10,index=ix[:10])}, c, "equal")
        assert abs(r.mean() - 0.5) < 0.01
    def test_sharpe_w(self, ix):
        from research.strategies.ensemble import ensemble_signals
        c = pd.Series(100 + np.cumsum(np.random.randn(50)*0.3), index=ix[:50])
        r = ensemble_signals({"a":pd.Series(np.random.choice([-1,0,1],50),index=ix[:50]),
                              "b":pd.Series(np.random.choice([-1,0,1],50),index=ix[:50])}, c, "sharpe_weight")
        assert not r.isna().all()
    def test_rank(self, ix):
        from research.strategies.ensemble import ensemble_signals
        c = pd.Series(100 + np.cumsum(np.random.randn(50)*0.3), index=ix[:50])
        r = ensemble_signals({"a":pd.Series([1]*50,index=ix[:50]), "b":pd.Series([0]*50,index=ix[:50]),
                              "c":pd.Series([-1]*50,index=ix[:50])}, c, "sharpe_rank")
        assert not r.isna().all()


class TestRegi:
    @pytest.fixture
    def d(self):
        np.random.seed(42)
        idx = pd.date_range("2024-01-01", periods=200, freq="5min")
        c = 100 + np.cumsum(np.random.randn(200)*0.3)
        return pd.DataFrame({"high":c+0.5,"low":c-0.5,"close":c}, index=idx)
    def test_sma(self, d):
        from research.strategies.regime import detect_regime_sma
        r = detect_regime_sma(d["close"])
        assert set(r.dropna().unique()).issubset({"bull","bear","sideways"})
    def test_adx(self, d):
        from research.strategies.regime import detect_regime_adx
        r = detect_regime_adx(d["high"],d["low"],d["close"])
        assert set(r.dropna().unique()).issubset({"bull","bear","sideways"})
    def test_filter(self, d):
        from research.strategies.regime import filter_by_regime
        s = pd.Series(np.random.choice([-1,0,1],200), index=d.index)
        reg = pd.Series("bear", index=d.index); reg.iloc[50:150]="bull"
        f = filter_by_regime(s, reg, allowed_regimes=["bull"])
        assert (f[reg=="bear"]==0).all()
