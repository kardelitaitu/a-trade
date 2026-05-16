# AGENTS.md — QuantumEdge Trading System

Binance 5-minute crypto trading system. Python-first prototyping.
Full context: `docs/`, `todo.md`, `CLI.md`, `FRAMEWORK.md`.

## Project Root
`C:\My Script\aaa-trade\`

## Conventions

- **Shell:** git-bash (MSYS) — POSIX syntax, `/c/...` paths
- **Python:** 3.11+, command is `python`
- **Venv:** `.venv` at root. Activate: `source .venv/Scripts/activate`
- **Git:** `feat:|fix:|chore:|docs:|test:|refactor:` commits. Feature branches off `main`. Repo: `kardelitaitu/a-trade`

## Documentation Reference

Before executing any task, check these documents:

| File | What it contains | When to reference |
|---|---|---|
| `CLI.md` | Every CLI command — data loading, backtesting, strategies, ML, optimization, paper trading, proxy, testing | Before writing shell commands or Python one-liners. Copy-paste ready. |
| `FRAMEWORK.md` | Full architecture — data flow, module docs, class interfaces, performance benchmarks, research findings | Before modifying any module. Understand the system before changing it. |
| `docs/strategy-guidelines.md` | Strategy interface, development process, naming conventions | When building new strategies |
| `docs/backtesting-standards.md` | Backtest methodology, metrics targets, acceptance criteria | When evaluating backtest results |
| `todo.md` | Current task list with checkboxes | After each task completion |

**Rule:** When the user asks to do something operational (run a backtest, fetch data, start paper trading), check `CLI.md` first. When they ask about architecture or modifying internals, check `FRAMEWORK.md` first.

## Mandatory Development Cycle

Every task follows this loop — no skipping steps:

1. **Plan** — present a clear plan with design options. Let user choose.
2. **Write todo** — break chosen plan into checklist items in `todo.md`
3. **Review todo** — show to user for confirmation before executing
4. **Execute todo** — build one item at a time, commit, check off, push

## Code Expectations

- Type hints + docstrings on all public functions
- `pytest` — 106 tests across all modules
- Vectorized where possible (pandas/numpy/vectorbt)
- Config via dict, never hardcoded numbers
- Numba `@jit` for state machine loops (backtest trade extraction, strategy signal generation, Monte Carlo batch kernels)
- Walk-forward validation, overfitting checks, realistic costs per backtesting-standards.md
- Before large computation: confirm with user first

## Hardware Optimization

- **CPU:** 32 threads — numba `@jit` and `prange` use all cores. Monte Carlo batch: 758 combos/sec.
- **RAM:** 96 GB DDR5 — LightGBM training uses 32 threads. Full indicator bank fits.
- **Storage:** 7000 MB/s NVMe SSD — Parquet I/O for 876K-row dataset takes <0.5s.
- Backtest throughput: **~0.3s per run on 876K rows**, **26ms per run on 100K rows**.

### Parallel Monte Carlo Workers

For `parallel_monte_carlo()`, use **n_jobs=24** (not 32). Running 32 Python worker processes simultaneously causes system instability due to OS scheduler contention. n_jobs=24 provides optimal throughput without stability issues.

| n_jobs | Stability | Speed | When to use |
|---|---|---|---|
| 8 | ✅ Stable | Moderate | Conservative, always works |
| **24** | ✅ **Stable** | **Fast** | **Default — best balance** |
| 32 | ❌ Unstable | Marginal gain | Avoid — scheduler thrashing |

## Project Architecture

```
research/
├── data/              Loader, fetcher, ProxyManager → Parquet
├── features/          Technical indicators + 35-feature ML matrix
├── strategies/        5 classical strategies + regime + ensemble (numba)
├── backtest/          Vectorized engine + 15+ metrics + batch numba kernels
├── ml/                LightGBM pipeline + feature importance
├── optimization/      Walk-forward, Monte Carlo (parallel), robustness
└── execution/         Paper trading pipeline (cron-ready)
```

See `FRAMEWORK.md` for detailed module documentation.

## Phase Status

All 4 phases complete. Key finding: RSI Mean Reversion (period=25, oversold=30, overbought=88) achieves **Sharpe +0.31, PF 1.12, DD -6.45%** on 2024-2025 out-of-sample.

## Quick Start

```bash
source .venv/Scripts/activate
python -m pytest -v              # Run all 106 tests
python -m pytest --cov=research  # Coverage report

# Run a strategy
python -c "
from research.data.loader import load_parquet
from research.strategies.mean_reversion import MeanReversion
from research.backtest.engine import VectorizedBacktest
from research.backtest.metrics import format_metrics_report
data = load_parquet()
strat = MeanReversion({'mode':'rsi','rsi_period':25,'rsi_oversold':30,'rsi_overbought':88})
signals = strat.generate_signals(data.loc['2024-01-01':'2025-12-31'])
result = VectorizedBacktest(data.loc['2024-01-01':'2025-12-31']).run(signals)
print(format_metrics_report(compute_metrics(result.equity_curve, result.trades)))
"

# Monte Carlo optimization (10K combos in ~13s)
python -c "
from research.data.loader import load_parquet
from research.optimization.monte_carlo import monte_carlo_ma, save_mc_report
from pathlib import Path
data = load_parquet()
r = monte_carlo_ma(data['close'].values.astype(float), (5, 50), (20, 100))
print(r.summary())
save_mc_report(r, Path('results/reports'))
"
```

See `CLI.md` for the complete command reference.
