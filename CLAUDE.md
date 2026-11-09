# AC-209A_VolForecast

Scaffold only: no data downloaded yet, Milestone 2 notebook stubbed.

## Scope

- Target: log Garman-Klass realized volatility over the next 5 trading days, per stock and week.
- Q1: pooled ridge/lasso and gradient-boosted trees vs per-stock HAR, MSE and QLIKE,
  Diebold-Mariano at 5%, test period 2019-2026 never used for tuning.
- Q2: which features account for any gain (permutation importance, SHAP), calm vs stress (2020, 2022).
- Optional if time permits: clustering of HAR coefficients, MLP/LSTM.

## Non-goals

- No portfolio backtest, Sharpe ratios, drawdowns or Fama-French excess-return regression
  (staff notes, approved projects list row 12).
- Not a data mirror: raw Yahoo prices are never committed.

## Invariants

- No look-ahead: every feature is known at the Friday close before the target week. XBRL facts
  are keyed to filing date, index membership to the date it was in force.
- Walk-forward splits with a 5-day gap between train and test windows.
- AC209A requirement: at least one method not covered in class (candidates: HAR/GARCH via `arch`,
  QLIKE with the Diebold-Mariano test); confirm with the TF.
- Notebooks present, `src/volforecast` computes.

## Layout

See README.md.

## Commands

```bash
uv sync
uv run ruff check .
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/00_eda.ipynb
```
