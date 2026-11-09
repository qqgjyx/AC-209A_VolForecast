# AC-209A_VolForecast

CS1090A / AC209A Fall 2026, project 12, Canvas group 38.
TF mentor: Daniel More Torres.

**Question.** For each S&P 500 stock and week, predict next-week realized volatility (log of a
Garman-Klass range estimator over the next 5 trading days). Out of sample, does a pooled model
with cross-sectional features (ridge/lasso, gradient-boosted trees) beat per-stock HAR on MSE and
QLIKE, which features account for any gain, and does the gain hold in both calm and stress
periods? The portfolio backtest from the original proposal is out of scope per the staff notes.

Group: Ashley Lee, Juntang Wang, Shuo Yang, Simon Manivong, Terry Su.

## Setup

```bash
uv sync
uv run jupyter lab
```

Raw data is not committed. The notebooks download it into `data/`; see `data/README.md` for
sources and licences.

## Layout

```
src/volforecast/   library code the notebooks import
notebooks/         00_eda.ipynb is the Milestone 2 notebook (get, understand, clean the data)
data/              downloaded data, gitignored except README.md
images/            figures for the slides and the revised proposal
out/               regenerable outputs, gitignored
```

## Milestones

| Milestone | Due | Deliverables |
|:--|:--|:--|
| MS2 data wrangling | Fri Nov 13, 9:59 pm ET (or the start of the TF presentation) | slides PDF, `notebooks/00_eda.ipynb`, revised proposal PDF |

## Dev

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/00_eda.ipynb
```

Work on a branch and open a pull request into `main`.
