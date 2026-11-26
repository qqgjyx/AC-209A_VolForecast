import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from volforecast import factors


def panel(n_weeks: int = 30, n_stocks: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weeks = np.repeat(pd.date_range("2010-01-01", periods=n_weeks, freq="W-FRI"), n_stocks)
    x = rng.normal(size=len(weeks))
    noise, z, e = rng.normal(size=(3, len(weeks)))
    return pd.DataFrame(
        {"week": weeks, "good": x, "copy": 2 * x + 1, "noise": noise, "y": x + 0.5 * e, "z": z}
    )


def test_rank_ic_is_spearman_within_each_week():
    p = panel()
    ic = factors.rank_ic(p, ["good", "noise"], "y")
    first = p[p["week"] == p["week"].iloc[0]]
    assert ic["good"].iloc[0] == pytest.approx(spearmanr(first["good"], first["y"])[0])
    s = factors.ic_summary(ic)
    assert s.index[0] == "good" and s.loc["good", "hit"] == 1.0 and s.loc["good", "icir"] > 3


def test_rank_ic_ranks_only_the_rows_where_both_are_known():
    p = panel()
    p.loc[p.index % 3 == 0, "good"] = np.nan
    ic = factors.rank_ic(p, ["good"], "y")
    first = p[p["week"] == p["week"].iloc[0]].dropna(subset=["good"])
    assert ic["good"].iloc[0] == pytest.approx(spearmanr(first["good"], first["y"])[0])


def test_clusters_group_a_factor_with_its_copy():
    labels = factors.clusters(panel(), ["good", "copy", "noise", "z"])
    assert labels["good"] == labels["copy"] and labels["noise"] != labels["good"]


def test_composite_points_factors_the_same_way():
    p = panel().assign(neg=lambda d: -d["good"])
    score = factors.composite(p, pd.Series({"good": 1.0, "neg": -1.0}))
    assert factors.rank_ic(p.assign(c=score), ["c"], "y")["c"].mean() > 0.7


def test_evolve_finds_a_planted_formula():
    rng = np.random.default_rng(0)
    p = pd.DataFrame({"week": np.repeat(np.arange(20), 100), "a": rng.normal(size=2000),
                      "b": rng.normal(size=2000)})  # fmt: skip
    p["y"] = p["a"] * p["b"] + 0.1 * rng.normal(size=2000)
    p.loc[:9, "y"] = np.nan  # fewer known targets than rows, and fewer rows than n_rows
    programs = factors.evolve(p, ["a", "b"], "y", rounds=1, population_size=200, generations=5,
                              n_jobs=1)  # fmt: skip
    fill = p[["a", "b"]].median()
    scored = factors.score_evolved(programs, {"train": p}, ["a", "b"], "y", fill)
    assert abs(scored["IC, train"].iloc[0]) > 0.8
    one = pd.DataFrame({"a": [np.nan], "b": [1.0]})  # a missing input gets the training median,
    more = pd.concat([one, pd.DataFrame({"a": [50.0, 90.0], "b": [0.0, 0.0]})])  # not this frame's
    first = [factors.evolved_factors(programs, f, ["a", "b"], fill).iloc[0, 0] for f in [one, more]]
    assert first[0] == first[1]
