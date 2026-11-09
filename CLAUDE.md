# AC-209A_VolForecast

Milestone 2 deliverables in progress: `ms2.ipynb` builds the stock-week table; slides and the revised
proposal live in `deliverables/ms2/`.

## Scope

- Target: log of next week's summed daily Garman-Klass variance, per stock and week (weeks end Friday).
- Q1: pooled ridge/lasso and gradient-boosted trees vs per-stock HAR, QLIKE and MSE,
  Diebold-Mariano at 5%, test period 2019-2026 never used for tuning.
- Q2: which feature blocks account for any gain, calm vs stress weeks (VIX >= 30 at the Friday close).
- Optional if time permits: clustering of HAR coefficients, MLP/LSTM (then Lightning + CSVLogger).

## Non-goals

- No portfolio backtest, Sharpe ratios, drawdowns or Fama-French excess-return regression
  (staff notes, approved projects list row 12).
- Not a data mirror: no Yahoo, Alpaca or WRDS data is ever committed.
- No qlib, Hydra, wandb or experiment tracker; results are files under `out/`.

## Invariants

- No look-ahead: every feature is known at the Friday close; `tests/` checks this by truncation.
  Cleaning rules read one row or the hand-built `RENAMES` / `WRONG_SERIES` tables, never later rows.
- Membership is the snapshot in force on each date; lags never cross a gap in a ticker's rows.
- AC209A outside method: HAR (Corsi 2009), QLIKE (Patton 2011), Diebold-Mariano (1995); confirm with the TF.
- The notebook presents, `src/volforecast` computes; one test file per module.

## Layout

See README.md.

## Commands

```bash
uv sync
uv run pytest && uv run ruff check . && uv run mypy
uv run python scripts/smoke_test.py --notebooks
uv run python scripts/build_rv5.py   # hours; needs Alpaca keys in .env
```
