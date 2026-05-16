# AGENTS.md — QuantumEdge Trading System

Binance 5-minute crypto trading system. Python-first prototyping.
Full context: `docs/`, `todo.md`.

## Project Root
`C:\My Script\aaa-trade\`

## Conventions

- **Shell:** git-bash (MSYS) — POSIX syntax, `/c/...` paths
- **Python:** 3.11+, command is `python`
- **Venv:** `.venv` at root. Activate: `source .venv/Scripts/activate`
- **Git:** `feat:|fix:|chore:|docs:|test:|refactor:` commits. Feature branches off `main`. Repo: `kardelitaitu/a-trade`

## Mandatory Development Cycle

Every task follows this loop — no skipping steps:

1. **Plan** — present a clear plan with design options. Let user choose.
2. **Write todo** — break chosen plan into checklist items in `todo.md`
3. **Review todo** — show to user for confirmation before executing
4. **Execute todo** — build one item at a time, commit, check off, push

## Code Expectations

- Type hints + docstrings on all public functions
- `pytest` — 96 tests across all modules
- Vectorized where possible (pandas/numpy/vectorbt)
- Config via dict, never hardcoded numbers
- Numba `@jit` for state machine loops (backtest trade extraction, strategy signal generation)
- Walk-forward validation, overfitting checks, realistic costs per backtesting-standards.md
- Before large computation: confirm with user first

## Hardware Optimization

- **CPU:** 32 threads — numba `@jit` accelerates trade extraction and strategy state machines
- **RAM:** 96 GB DDR5 — LightGBM training uses 32 threads
- **Storage:** 7000 MB/s NVMe SSD — Parquet I/O for 876K-row dataset
- Backtest throughput: **~0.3s per run on 876K rows**, **26ms per run on 100K rows**

## Project Architecture

```
research/
├── data/              Loader, fetcher, cleaning → Parquet
├── features/          Technical indicators + 35-feature ML matrix
├── strategies/        5 classical strategies + regime + ensemble all numba-accelerated
├── backtest/          Vectorized engine + 15+ metrics (numba trade extraction)
├── ml/                LightGBM pipeline + feature importance
├── optimization/      Random search, walk-forward, Monte Carlo, robustness
└── execution/         Paper trading pipeline (cron-ready)
```

## Phase Status

All 4 phases complete. Key finding: RSI Mean Reversion (period=25, oversold=30, overbought=88) achieves **Sharpe +0.31, PF 1.12, DD -6.45%** on 2024-2025 out-of-sample.

## Quick Commands

```
source .venv/Scripts/activate
python -m pytest -v              # Run all 96 tests
python -m pytest --cov=research  # Coverage report
python research/execution/paper.py  # Run paper trading
```
