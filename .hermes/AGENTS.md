# AGENTS.md — QuantumEdge Trading System

## Project Identity

**QuantumEdge** — a data-driven 5-minute cryptocurrency trading system targeting Binance spot/futures markets. Python-first prototyping with professional architecture. See `whitepaper.md`, `roadmap.md`, `profitable-strategy.md` for full context.

## Project Root

`C:\My Script\aaa-trade\`

## Base Conventions

- **Paths**: use forward-slash MSYS style (`/c/My Script/aaa-trade/`) in shell commands.
- **Python**: 3.11+. Use `python` (not `python3`) on this Windows host.
- **Venv**: use `.venv` at project root for the virtual environment. Activate: `source .venv/Scripts/activate` (git-bash on Windows).
- **Git**: commits with conventional commits format (`feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`). Branch strategy: feature branches off `main`.

## Project Structure

Based on whitepaper.md — implement this layout as the project grows:

```
research/
├── data/           # Fetchers, loaders, data cleaning
├── features/       # Technical indicators, ML features
├── strategies/     # Strategy classes (rule-based / ML / ensemble)
├── backtest/       # Engine, metrics, visualization
├── optimization/   # Walk-forward, hyperparameter tuning
├── ml/             # Models, regime detection
├── risk/           # Position sizing, stops, portfolio
└── notebooks/      # Jupyter research notebooks
execution/          # Future: Rust/Go live engine (optional)
shared/             # Data models, enums, communication
config/             # Configuration files (YAML/TOML)
results/            # Backtest reports, equity curves, logs
```

## How I Operate

### Process
1. **Plan first** — before writing significant code, present a clear plan with design options. Let the user choose.
2. **Incremental delivery** — build one module at a time. Commit working code before moving on.
3. **Test coverage** — every feature comes with tests. Use `pytest` (with `pytest-cov` for coverage).
4. **Stats rigor** — validate statistical significance, avoid overfitting. Walk-forward, OOS testing, Monte Carlo.

### Communication
- Present plans as structured lists or bullet points.
- Flag risks, edge cases, and statistical pitfalls proactively.
- Use data over opinion — show numbers, metrics, distributions.
- Before heavy computation, confirm with user.

### Code Quality
- Type hints on all Python functions.
- Docstrings for public API and non-trivial logic.
- Keep functions small and single-purpose.
- Prefer `pandas`, `numpy`, `vectorbt` / `backtesting.py` for vectorized work.
- Configuration via YAML/TOML, not hardcoded magic numbers.

### Memory & Skills
- Save durable project conventions, data quirks, tool versions, and preferred libraries to memory.
- Save reusable workflows (e.g., "how to backtest a strategy end-to-end") as skills.
- Do NOT log session task progress to memory.

## Phase Order (from roadmap.md)

1. **Foundation** — Data downloader & storage, EDA, core backtest engine
2. **Classical Strategies** — MA/breakout, mean reversion, volatility, regime detection
3. **Advanced Models** — ML models, feature importance, ensemble
4. **Validation & Deployment** — Rigorous validation, paper trading, live, optional Rust/Go migration

## Useful Commands

```
source .venv/Scripts/activate   # Activate venv (git-bash)
python -m pytest -v             # Run tests
python -m pytest --cov=research # Coverage report
pip install -e .                # Editable install (when setup.py/pyproject.toml exists)
```
