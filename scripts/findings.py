"""Two checks behind the target and the stress label, written for the slides.

Reads only the cached membership, Yahoo prices and FRED VIX (the notebook's own caches under
data/, so it downloads nothing the notebook would not), and needs no API keys. Writes two figures
to images/ and the macros the slides quote to deliverables/ms2/findings.tex.

Usage:
    uv run python scripts/findings.py

Checks against the notebook: the companies with Yahoo prices, the weeks and the VIX >= 30 stress
weeks must equal the notebook's own \\nYahoo, \\nWeeks and \\nStressWeeks, or the script stops.
Alpaca's daily bars are not read, so the variance shares cover the companies with Yahoo prices.
"""

import re
import sys
from pathlib import Path

import pandas as pd

from volforecast import diagnostics as dg
from volforecast import plots
from volforecast.clean import clean
from volforecast.data import (
    BREAKS,
    WRONG_SERIES,
    fetch_fred,
    fetch_membership,
    fetch_prices,
    to_current,
    universe,
)
from volforecast.features import daily_measures, week_end

ROOT = Path(__file__).resolve().parents[1]
DATA, IMAGES, MS2 = ROOT / "data", ROOT / "images", ROOT / "deliverables" / "ms2"
START, END = "2005-01-01", "2026-11-07"  # the notebook's dates; END is exclusive
# Variance shares use the train and tune years only: the test years must not shape the target.
SHARE_START, SHARE_END = dg.SPLITS["train"][0], dg.SPLITS["tune"][1]
EARLY_END, LATE_START = "2009-12-31", "2010-01-01"  # the shares before and after the crisis years
STRESS_YEAR_WEEKS = 8  # slide 9 names each year with at least this many VIX >= 30 weeks


def notebook_macro(name: str) -> str:
    """A value from the notebook's numbers.tex, for the cross-checks."""
    text = (MS2 / "numbers.tex").read_text()
    match = re.search(rf"\\newcommand{{\\{name}}}{{([^}}]*)}}", text)
    if match is None:
        raise KeyError(name)
    return match.group(1)


def pct(x: float) -> str:
    """A share as a whole percent for LaTeX."""
    return f"{x * 100:.0f}\\%"


def years(start: str, end: str, short_end: bool = False) -> str:
    """A span of years for LaTeX, such as 2005--2018, or 2005--09 with `short_end`."""
    first, last = start[:4], end[:4]
    return f"{first}--{last[2:] if short_end else last}"


def year_list(chosen: list[int]) -> str:
    """Years in words, with runs joined: [2008, 2009, 2011, 2020] -> 2008--09, 2011 and~2020."""
    runs: list[list[int]] = []
    for y in sorted(chosen):
        if runs and y == runs[-1][-1] + 1:
            runs[-1].append(y)
        else:
            runs.append([y])
    parts = [f"{r[0]}--{str(r[-1])[2:]}" if len(r) > 1 else str(r[0]) for r in runs]
    # "and~" keeps the last year on the line with "and" (a LaTeX non-breaking space).
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and~" + parts[-1]


def main() -> int:
    membership = to_current(fetch_membership(DATA / "sp500_membership.csv"))
    tickers = universe(membership, START)
    prices = fetch_prices(tickers, START, END, DATA / "prices")
    vix = fetch_fred(["VIXCLS"], DATA / "fred")["VIXCLS"].loc[START:]
    breaks = {t: d for t, (d, _) in BREAKS.items()}
    daily, _ = clean(prices.reset_index(), membership, set(WRONG_SERIES), breaks, END)
    measures = daily_measures(daily)

    # 1. What the daily range sees, per company, over the train and tune years. The early and
    # late shares compare the same companies, so index turnover can't pass for a change over time.
    shares = dg.variance_shares(measures, SHARE_START, SHARE_END)
    early = dg.variance_shares(measures, SHARE_START, EARLY_END)
    late = dg.variance_shares(measures, LATE_START, SHARE_END)
    in_both = early.index.intersection(late.index)
    early, late = early.loc[in_both], late.loc[in_both]

    # 2. Stress weeks per split, on the weeks the modeling table has (the last one has no target).
    weeks = pd.DatetimeIndex(week_end(daily["date"]).unique()).sort_values()[:-1]
    vix_weeks = dg.weekly_vix(vix).reindex(weeks)
    by_split = {t: dg.stress_by_split(vix_weeks, t) for t in (30, 25)}
    by_year = dg.stress_weeks_by_year(vix_weeks, [30, 25])
    stress_years = by_year.index[by_year["VIX >= 30"] >= STRESS_YEAR_WEEKS]

    # Check against the notebook before writing anything, so a failed check leaves every output
    # (both figures and findings.tex) as it was.
    n_yahoo = len(set(tickers) & set(prices.index.unique("ticker")))  # depends on the price cache
    n_weeks, n_stress = len(weeks), int(by_split[30]["stress_weeks"].sum())
    found = n_yahoo, n_weeks, n_stress
    expected = tuple(
        int(notebook_macro(m).replace(",", "")) for m in ("nYahoo", "nWeeks", "nStressWeeks")
    )
    if found != expected:
        print(f"MISMATCH with the notebook: {found} (want {expected})")
        print("Nothing was written.")
        return 1

    names = ["Garman-Klass", "Garman-Klass + overnight"]
    plots.variance_coverage(
        shares[["gk", "gk_overnight"]].set_axis(names, axis=1), IMAGES / "variance_coverage.png"
    )
    labels = {c: c.replace(">=", "≥") for c in by_year.columns}
    spans = {s: (int(a[:4]), int(b[:4])) for s, (a, b) in dg.SPLITS.items() if s != "train"}
    plots.stress_weeks(by_year.rename(columns=labels), spans, IMAGES / "stress_weeks.png")

    numbers = {
        "shareCompanies": f"{len(shares):,}",
        "shareCompaniesBoth": f"{len(in_both):,}",
        "shareYears": years(SHARE_START, SHARE_END),
        "shareYearsEarly": years(SHARE_START, EARLY_END, short_end=True),
        "shareYearsLate": years(LATE_START, SHARE_END, short_end=True),
        "overnightShare": pct(shares["overnight"].median()),
        "overnightShareLow": pct(shares["overnight"].quantile(0.1)),
        "overnightShareHigh": pct(shares["overnight"].quantile(0.9)),
        "overnightShareEarly": pct(early["overnight"].median()),
        "overnightShareLate": pct(late["overnight"].median()),
        "gkShare": pct(shares["gk"].median()),
        "gkOvernightShare": pct(shares["gk_overnight"].median()),
        "stressTestTwenty": f"{int(by_year.loc[2020, 'VIX >= 30'])}",
        "stressYears": year_list(list(stress_years)),
    }
    for split in dg.SPLITS:  # \\trainWeeks, \\stressTrain, \\stressTrainEpisodes, ...
        s30, s25, cap = by_split[30].loc[split], by_split[25].loc[split], split.capitalize()
        numbers[f"{split}Weeks"] = f"{int(s30['weeks'])}"
        numbers[f"stress{cap}"] = f"{int(s30['stress_weeks'])}"
        numbers[f"stress{cap}Episodes"] = f"{int(s30['episodes'])}"
        numbers[f"stress{cap}Low"] = f"{int(s25['stress_weeks'])}"
    tex = "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in numbers.items())
    header = "% Written by scripts/findings.py; do not edit.\n"
    (MS2 / "findings.tex").write_text(header + tex + "\n")
    print(tex)
    print(f"\nYahoo companies {n_yahoo}, weeks {n_weeks:,}, VIX >= 30 weeks {n_stress}: all match")
    print(by_split[30].to_string(), by_split[25].to_string(), sep="\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
