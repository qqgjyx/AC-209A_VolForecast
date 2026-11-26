# AC-209A_VolForecast

Milestone 3 drafted: `ms3.ipynb` (EDA, factors, baselines, ridge/lasso on validation 2016-2018,
pipeline) and `deliverables/ms3/`; `ms2.ipynb` builds the stock-week table. Final models are next.

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
- Every model is a scikit-learn regressor on table rows and runs through `pipeline.walk_forward`
  (refit at each period start on rows whose target week ended before it); test years 2019-2026 are
  read once, for the final evaluation. Validation (2016-2018) chooses penalties and blocks.
  A row whose target week ends in 2019 or later is a test row (`pipeline.split_of`), and ms3.ipynb
  blanks every test target on load. FRED's H.15 rates enter a day late (`pipeline.NEXT_DAY`).
- EDA that informs a choice reads the training years (2005-2015) only.

## Layout

See README.md.

## Commands

```bash
uv sync
uv run pytest && uv run ruff check . && uv run mypy
uv run python scripts/smoke_test.py --notebooks
uv run python scripts/build_rv5.py   # hours; needs Alpaca keys in .env
```
