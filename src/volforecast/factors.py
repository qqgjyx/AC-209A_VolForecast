"""Factor research as a quant desk runs it, applied to volatility instead of returns.

Each stock-level feature is treated as a factor and scored by its rank information coefficient
(IC): the Spearman correlation, across the stocks of one week, between the factor and what it
should predict, summarized over weeks by its mean, ICIR (mean / std) and hit rate, as in qlib
and alphalens. Scoring against HAR's miss (the residual IC) measures what a factor adds beyond
HAR, the way a desk neutralizes a factor against its risk model. Then: how the IC evolves by
year and regime, how factors cluster, an IC-weighted multi-factor composite, and a small
genetic-programming search for new factors (gplearn), the classic form of factor evolution.
"""

import numpy as np
import pandas as pd
from gplearn._program import _Program
from gplearn.genetic import SymbolicTransformer
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def rank_ic(frame: pd.DataFrame, factors: list[str], target: str, by: str = "week") -> pd.DataFrame:
    """Rank IC: Spearman correlation across stocks in each period (rows), one column per factor.

    Each factor is paired with the target on the rows where both are known, and both are ranked
    on those rows only, so a missing value cannot shift the other ranks. A factor that is
    constant across stocks in a period (VIX) has no IC.
    """
    known = frame[factors].notna() & frame[target].notna().to_numpy()[:, None]
    x = frame[factors].where(known)
    y = pd.DataFrame(dict.fromkeys(factors, frame[target])).where(known)
    rx, ry = x.groupby(frame[by]).rank(), y.groupby(frame[by]).rank()
    return rx.groupby(frame[by]).apply(lambda g: g.corrwith(ry.loc[g.index]))


def ic_summary(ic: pd.DataFrame) -> pd.DataFrame:
    """Mean IC, its standard deviation, ICIR (mean / std), t-statistic and hit rate per factor.

    The hit rate is the share of periods whose IC has the sign of the mean IC.
    """
    mean, std, n = ic.mean(), ic.std(), ic.notna().sum()
    hit = (np.sign(ic) == np.sign(mean)).sum() / n
    out = pd.DataFrame({"ic": mean, "ic_std": std, "icir": mean / std,
                        "t": mean / std * np.sqrt(n), "hit": hit})  # fmt: skip
    return out.sort_values("ic", key=np.abs, ascending=False)


def clusters(frame: pd.DataFrame, factors: list[str], threshold: float = 0.3,
             n_rows: int = 20_000, seed: int = 0) -> pd.Series:  # fmt: skip
    """Group factors whose values move together: average-linkage clustering on 1 - |Spearman|.

    Two factors in one cluster have |rank correlation| above 1 - `threshold` on average, so a
    cluster is one signal measured several ways. Uses a random sample of `n_rows` rows.
    """
    sample = frame[factors].sample(min(n_rows, len(frame)), random_state=seed)
    corr = sample.rank().corr().abs().fillna(0).to_numpy()  # Spearman: Pearson on ranks
    dist = squareform(1 - corr, checks=False).clip(min=0)
    labels = fcluster(linkage(dist, method="average"), t=threshold, criterion="distance")
    return pd.Series(labels, index=factors, name="cluster")


def composite(frame: pd.DataFrame, weights: pd.Series, by: str = "week") -> pd.Series:
    """A multi-factor score: the weighted sum of each factor's cross-sectional z-score.

    Weights carry the sign of each factor's IC (an IC-weighted composite); a missing value
    counts as the period's mean (z-score 0).
    """
    x = frame[list(weights.index)]
    z = (x - x.groupby(frame[by]).transform("mean")) / x.groupby(frame[by]).transform("std")
    return (z.fillna(0) * weights).sum(axis=1).rename("composite")


def evolve(frame: pd.DataFrame, inputs: list[str], target: str, rounds: int = 3,
           n_rows: int = 30_000, seed: int = 0, **gp: object) -> list[_Program]:  # fmt: skip
    """Search formulas of `inputs` that rank-correlate with `target`, by genetic programming.

    Each round, gplearn's SymbolicTransformer evolves a population of formulas over generations
    and keeps the one with the highest Spearman correlation to the target, with a penalty on
    length; the next round searches what that formula leaves unexplained (the target minus its
    least-squares fit on the formulas found so far), so the factors found are distinct, as an
    evolving factor library requires. Fit on a random sample of `n_rows` rows; missing inputs
    are set to the median of `frame`, which `evolved_factors` must be given to score the formulas
    the same way. Extra keyword arguments go to SymbolicTransformer.
    """
    rows = frame.dropna(subset=[target])
    sample = rows.sample(min(n_rows, len(rows)), random_state=seed)
    x = sample[inputs].fillna(frame[inputs].median()).to_numpy()
    y, left = sample[target].to_numpy(), sample[target].to_numpy()
    params: dict[str, object] = {
        "population_size": 600, "generations": 6, "hall_of_fame": 20, "n_components": 1,
        "function_set": ("add", "sub", "mul", "div", "log", "sqrt", "abs", "neg", "max", "min"),
        "metric": "spearman", "parsimony_coefficient": 0.002, "max_samples": 0.8,
        "feature_names": inputs, "n_jobs": -1,  # gplearn seeds each program: same result
    } | gp  # fmt: skip
    found: list[_Program] = []
    for k in range(rounds):
        search = SymbolicTransformer(random_state=seed + k, **params).fit(x, left)
        found.append(search[0])
        values = np.nan_to_num(np.column_stack([np.ones(len(x))] + [p.execute(x) for p in found]))
        left = y - values @ np.linalg.lstsq(values, y, rcond=None)[0]
    return found


def evolved_factors(
    programs: list[_Program], frame: pd.DataFrame, inputs: list[str], fill: pd.Series
) -> pd.DataFrame:
    """Values of each evolved formula on `frame`, one column per formula (named f1, f2, ...).

    fill: the value for a missing input, the training medians `evolve` used, so a later frame
    is scored with the same formulas and no statistic of its own.
    """
    x = frame[inputs].fillna(fill).to_numpy()
    return pd.DataFrame(
        {f"f{i + 1}": p.execute(x) for i, p in enumerate(programs)}, index=frame.index
    )


def formulas(programs: list[_Program]) -> pd.Series:
    """The evolved formulas, as text, in the order of `evolved_factors`' columns."""
    return pd.Series([str(p) for p in programs], index=[f"f{i + 1}" for i in range(len(programs))],
                     name="formula")  # fmt: skip


def score_evolved(programs: list[_Program], frames: dict[str, pd.DataFrame], inputs: list[str],
                  target: str, fill: pd.Series, by: str = "week") -> pd.DataFrame:  # fmt: skip
    """Each evolved formula with its mean rank IC and ICIR against `target` in each frame.

    frames: e.g. {"train": ..., "validation": ...}; a formula found on the training years that
    keeps its IC on the validation years has not just fit noise. fill: as in `evolved_factors`.
    """
    out = {}
    for name, frame in frames.items():
        values = evolved_factors(programs, frame, inputs, fill)
        ic = rank_ic(pd.concat([frame[[by, target]], values], axis=1), list(values), target, by)
        summary = ic_summary(ic)
        out[f"IC, {name}"], out[f"ICIR, {name}"] = summary["ic"], summary["icir"]
    return pd.concat([formulas(programs), pd.DataFrame(out)], axis=1)
