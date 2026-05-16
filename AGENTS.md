# AGENTS.md — QuantumEdge Trading System

Binance 5-minute crypto trading system. Python-first prototyping → optional Rust/Go.
Full context: `docs/whitepaper.md`, `docs/roadmap.md`, `docs/strategy-guidelines.md`, `docs/backtesting-standards.md`.

## Project Root
`C:\My Script\aaa-trade\`

## Conventions

- **Shell:** git-bash (MSYS) — POSIX syntax, `/c/...` paths
- **Python:** 3.11+, command is `python` (not `python3`)
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
- `pytest` + `pytest-cov` for every feature
- Vectorized where possible (pandas/numpy/vectorbt)
- Config via YAML/TOML, never hardcoded numbers
- Walk-forward validation, overfitting checks, realistic costs per backtesting-standards.md
- Before large computation: confirm with user first

## Skill Workflows

Save reusable workflows (end-to-end backtest, data fetch, etc.) as skills. Do NOT log task progress to memory.

## Phase Reference

- **Phase 1 (current):** Data downloader, EDA, core backtest engine
- **Phase 2:** Classical strategies (MA, mean reversion, volatility, regime)
- **Phase 3:** ML models, feature importance, ensemble
- **Phase 4:** Validation, paper trading, live, Rust/Go migration

## Quick Commands

```
source .venv/Scripts/activate
python -m pytest -v
python -m pytest --cov=research
```

## Hardware Optimization

- **CPU:** 32 threads
- **RAM:** 96 GB DDR5
- **Storage:** 7000 MB/s NVMe SSD
- Code should be optimized to fully utilize this hardware. High CPU, RAM, and SSD usage is fine.
- Prefer parallelized/vectorized operations (numba, multiprocessing, batch processing) over single-threaded loops.
- Backtest throughput priority: faster ops/sec is the goal, not memory conservation.