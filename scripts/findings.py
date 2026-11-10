"""Two checks behind the target and the stress label, written for the slides.

Reads only the cached membership, Yahoo prices and FRED VIX (the notebook's own caches under
data/, so it downloads nothing the notebook would not), and needs no API keys. Writes two figures
to images/ and the macros the slides quote to deliverables/ms2/findings.tex.

Usage:
    uv run python scripts/findings.py

Checks against the notebook: the weeks and the VIX >= 30 stress weeks it counts must equal the
notebook's own \\nWeeks and \\nStressWeeks, or the script stops.
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
SHARE_END = "2018-12-31"  # train and tune years only: the test years must not shape the target


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


def main() -> int:
    membership = to_current(fetch_membership(DATA / "sp500_membership.csv"))
    tickers = universe(membership, START)
    prices = fetch_prices(tickers, START, END, DATA / "prices")
    vix = fetch_fred(["VIXCLS"], DATA / "fred")["VIXCLS"].loc[START:]
    breaks = {t: d for t, (d, _) in BREAKS.items()}
    daily, _ = clean(prices.reset_index(), membership, set(WRONG_SERIES), breaks, END)
    measures = daily_measures(daily)

    # 1. What the daily range sees, per company, over the train and tune years.
    shares = dg.variance_shares(measures, START, SHARE_END)
    early = dg.variance_shares(measures, START, "2009-12-31")
    late = dg.variance_shares(measures, "2010-01-01", SHARE_END)

    # 2. Stress weeks per split, on the weeks the modeling table has (the last one has no target).
    weeks = pd.DatetimeIndex(week_end(daily["date"]).unique()).sort_values()[:-1]
    vix_weeks = dg.weekly_vix(vix).reindex(weeks)
    by_split = {t: dg.stress_by_split(vix_weeks, t) for t in (30, 25)}
    by_year = dg.stress_weeks_by_year(vix_weeks, [30, 25])

    # Check against the notebook before writing anything, so a failed check leaves every output
    # (both figures and findings.tex) as it was.
    n_weeks, n_stress = len(weeks), int(by_split[30]["stress_weeks"].sum())
    expected = int(notebook_macro("nWeeks").replace(",", "")), int(notebook_macro("nStressWeeks"))
    if (n_weeks, n_stress) != expected:
        print(f"MISMATCH with the notebook: {n_weeks} weeks, {n_stress} stress (want {expected})")
        print("Nothing was written.")
        return 1

    names = ["Garman-Klass", "Garman-Klass + overnight"]
    plots.variance_coverage(
        shares[["gk", "gk_overnight"]].set_axis(names, axis=1), IMAGES / "variance_coverage.png"
    )
    labels = {c: c.replace(">=", "≥") for c in by_year.columns}
    plots.stress_weeks(by_year.rename(columns=labels), {"tune": (2016, 2018), "test": (2019, 2026)},
                       IMAGES / "stress_weeks.png")  # fmt: skip

    numbers = {
        "shareCompanies": f"{len(shares):,}",
        "overnightShare": pct(shares["overnight"].median()),
        "overnightShareLow": pct(shares["overnight"].quantile(0.1)),
        "overnightShareHigh": pct(shares["overnight"].quantile(0.9)),
        "overnightShareEarly": pct(early["overnight"].median()),
        "overnightShareLate": pct(late["overnight"].median()),
        "gkShare": pct(shares["gk"].median()),
        "gkOvernightShare": pct(shares["gk_overnight"].median()),
        "stressTestTwenty": f"{int(by_year.loc[2020, 'VIX >= 30'])}",
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
    print(f"\nweeks {n_weeks:,}, VIX >= 30 weeks {n_stress} (both match the notebook)")
    print(by_split[30].to_string(), by_split[25].to_string(), sep="\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
