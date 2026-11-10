# Follow-up analyses

Six analyses I first ran on 25 current stocks with daily rows (2010 onward), rerun on our stock-week table. They are reference material for Milestone 3: nothing in `src/`, the tests or the MS2 deliverables depends on this folder.

| File | What it is |
|---|---|
| `prior_findings.ipynb` | the six analyses, with a Takeaway after each |
| `prior_findings.py` | the functions the notebook calls |
| `test_prior_findings.py` | tests for those functions, on synthetic data with known answers |
| `findings.md` | the results next to the earlier ones, written by the notebook |
| `figures/` | one figure per analysis, written by the notebook |

The six analyses:

1. HAR fitted company by company, and pooled HAR with and without a constant per company
2. HAR's miss in calm and stress weeks
3. Next week's volatility after calm and stress weeks
4. HAR's miss by this week's return (the leverage effect)
5. How long volatility remembers, lags 1 to 52 weeks
6. How much stocks move together in calm and stress weeks

## How they were run

- Data: the MS2 notebook's caches under `data/` (membership, Yahoo prices, FRED VIX), cleaned with the same `clean()` call. No API keys. Alpaca's daily bars start in 2016, so leaving them out doesn't change the training years.
- Rows: stock-weeks in the index with a target, as in `features.build_table`, keeping only the columns these analyses need (target, `har_d`, `har_w`, `har_m`, `ret`, weekly VIX).
- Years: the training years only (2005 to 2015), since each analysis relates something known at a Friday close to next week's target.
- Stress: VIX at or above 30 at the week's last close, the same rule as the table.
- HAR fits need at least 104 weeks per company. All fits are in sample.

The earlier numbers in `findings.md` are not directly comparable: daily rows, 25 survivors, a close-to-close target and stress at VIX 25. Analysis 2's earlier number was out of sample on 2019 to 2026, and the new one is in sample on 2005 to 2015.

## Rerun

```bash
uv run jupyter nbconvert --to notebook --execute --inplace analysis/prior_findings.ipynb
uv run pytest analysis
```
