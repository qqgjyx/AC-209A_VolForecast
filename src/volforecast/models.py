"""Baselines and first pooled models, each a scikit-learn regressor on the stock-week table.

Every model takes the table's rows as a DataFrame (`ticker` and feature columns) and the target,
so `pipeline.walk_forward` refits any of them the same way. The baselines follow the three
training schemes of Zhang et al. (2024): one model per stock, one pooled model, and the pooled
model with market volatility added.
"""

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModel
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

from volforecast.alpha158 import alpha158_names

HAR = ["har_d", "har_w", "har_m"]
MARKET = ["spy_har_w", "VIXCLS"]  # market variance and implied volatility at the Friday close
BLOCKS = {  # the feature blocks of the revised question, in the order they are added
    "HAR": HAR,
    "long memory": ["har_q", "har_y"],
    "range": ["log_rv_parkinson", "log_rv_rs", "log_rv_overnight", "log_rv_neg"],
    "returns": ["ret", "ret_4w"],
    "liquidity": ["log_dollar_volume", "illiq"],
    "market": ["spy_har_w", "spy_ret", "VIXCLS", "VXVCLS", "BAA10Y", "T10Y2Y", "DTB3"],
    "earnings": ["weeks_since_earnings", "earnings_next_week"],
    "calendar": ["n_days"],
    "Alpha158": alpha158_names(),
    "industry": ["sic2"],
}
MIN_WEEKS = 52  # a year of a company's own rows before it gets its own HAR fit
GARCH_MIN_DAYS = 500  # two years of daily returns before a company gets its own GARCH fit
BACKCAST_DAYS = 75  # arch starts each variance path from its first 75 squared residuals
MIN_RETURNS = 10  # fewer daily returns than this get no GARCH forecast (the training mean)


class LastWeek(BaseEstimator, RegressorMixin):
    """No-change forecast: next week's log variance equals this week's (`har_w`)."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LastWeek":  # noqa: D102
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: D102
        return X["har_w"].to_numpy()


class StockMean(BaseEstimator, RegressorMixin):
    """The mean model: each company's training mean; a new company gets the overall mean."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StockMean":  # noqa: D102
        y = pd.Series(np.asarray(y), index=X.index)
        self.means_, self.overall_ = y.groupby(X["ticker"]).mean(), float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: D102
        return X["ticker"].map(self.means_).fillna(self.overall_).to_numpy(dtype=float)


def pooled_ols(columns: list[str]) -> Pipeline:
    """One least-squares regression on `columns` for all companies (pooled HAR, HAR + market)."""
    return make_pipeline(make_column_transformer(("passthrough", columns)), LinearRegression())


class PerStock(BaseEstimator, RegressorMixin):
    """One least-squares fit per company on `columns`: the per-stock HAR of Corsi (2009).

    Companies with fewer than `min_weeks` training rows, and new ones, use the pooled fit.
    """

    def __init__(self, columns: list[str] = HAR, min_weeks: int = MIN_WEEKS) -> None:  # noqa: D107
        self.columns, self.min_weeks = columns, min_weeks

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PerStock":  # noqa: D102
        x, y = X[self.columns].to_numpy(), np.asarray(y)
        self.pooled_ = LinearRegression().fit(x, y)
        coefs = {}
        for ticker, rows in X.groupby("ticker").indices.items():
            if len(rows) >= self.min_weeks:
                own = LinearRegression().fit(x[rows], y[rows])
                coefs[ticker] = [own.intercept_, *own.coef_]
        self.coef_ = pd.DataFrame.from_dict(coefs, orient="index", columns=["const", *self.columns])
        return self

    def own_fit(self, X: pd.DataFrame) -> np.ndarray:
        """True for the rows a company's own fit forecasts, False for the pooled fallback."""
        return X["ticker"].isin(self.coef_.index).to_numpy()

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: D102
        x = X[self.columns].to_numpy()
        c = self.coef_.reindex(X["ticker"]).to_numpy()
        own = c[:, 0] + np.einsum("ij,ij->i", c[:, 1:], x)
        return np.where(self.own_fit(X), own, self.pooled_.predict(x))


class FixedEffects(BaseEstimator, RegressorMixin):
    """Pooled slopes with a constant per company (the within estimator).

    Slopes come from the regression of each company's demeaned target on its demeaned inputs;
    a new company gets the pooled constant, mean(y) - mean(x) @ slopes over all training rows.
    """

    def __init__(self, columns: list[str] = HAR) -> None:  # noqa: D107
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FixedEffects":  # noqa: D102
        frame = X[self.columns].assign(y=np.asarray(y))
        means = frame.groupby(X["ticker"]).mean()
        within = frame - means.reindex(X["ticker"]).to_numpy()
        fit = LinearRegression(fit_intercept=False).fit(within[self.columns], within["y"])
        self.slopes_ = fit.coef_
        self.const_ = means["y"] - means[self.columns].to_numpy() @ self.slopes_
        self.default_ = float(frame["y"].mean() - frame[self.columns].mean().to_numpy() @ fit.coef_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: D102
        const = X["ticker"].map(self.const_).fillna(self.default_).to_numpy(dtype=float)
        return const + X[self.columns].to_numpy() @ self.slopes_


def linear(estimator: RegressorMixin, steady: list[str], heavy: list[str],
           categorical: list[str]) -> Pipeline:  # fmt: skip
    """A regularized linear model on every feature, with preprocessing fit on training rows only.

    steady: near-normal columns (the log variances), standardized; heavy: heavy-tailed columns
    (volume ratios, Alpha158), mapped to normal scores by rank, so an outlier cannot dominate;
    both then impute 0 (the training mean or median) and add a missing-value flag.
    categorical: one-hot, with unseen and missing values as all zeros.
    """
    impute = {"strategy": "constant", "fill_value": 0.0, "add_indicator": True}
    normal = QuantileTransformer(output_distribution="normal", subsample=200_000, random_state=0)
    pre = make_column_transformer(
        (make_pipeline(StandardScaler(), SimpleImputer(**impute)), steady),
        (make_pipeline(normal, SimpleImputer(**impute)), heavy),
        (OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float), categorical),
    )
    return make_pipeline(pre, estimator)


def _garch_spec(returns: np.ndarray) -> ARCHModel:
    """GARCH(1,1) with a constant mean on daily percent returns."""
    return arch_model(returns, mean="constant", vol="garch", p=1, q=1, rescale=False)


def _garch_params(returns: np.ndarray) -> np.ndarray:
    """Maximum-likelihood GARCH(1,1) parameters: [mu, omega, alpha, beta]."""
    return _garch_spec(returns).fit(disp="off", options={"maxiter": 200}).params.to_numpy()


def _garch_path(returns: pd.Series, params: np.ndarray) -> pd.Series:
    """Sum of the next five daily variance forecasts made on each day of `returns` (decimal)."""
    fixed = _garch_spec(returns.to_numpy()).fix(params)
    variance = fixed.forecast(horizon=5, start=0, reindex=False).variance
    return pd.Series(variance.to_numpy().sum(axis=1) / 1e4, index=returns.index[-len(variance) :])


def _garch_log_week_var(
    returns: pd.Series, params: np.ndarray, fridays: pd.DatetimeIndex
) -> np.ndarray:
    """Log of the sum of the next five daily variance forecasts made at each Friday close.

    The variance path is filtered with fixed `params` over all of `returns` (percent). arch
    starts the path from a weighted mean of the first 75 squared residuals, so a forecast made
    in a company's first 75 trading days is redone on the returns up to that day: every
    forecast uses that day's return and earlier ones only. Each Friday takes the forecast made
    at the company's last trading day of that week; units are decimal.
    """
    path = _garch_path(returns, params)
    at = pd.Series(np.arange(len(path)), index=path.index)
    at = at.reindex(fridays, method="ffill", tolerance=pd.Timedelta(days=6)).to_numpy()
    week_var = np.full(len(fridays), np.nan)
    known = ~np.isnan(at)
    week_var[known] = path.to_numpy()[at[known].astype(int)]
    early = min(BACKCAST_DAYS, len(returns)) - 1  # forecasts before this day read later returns
    for k in np.flatnonzero(known & (at < early)):
        n = int(at[k]) + 1
        week_var[k] = _garch_path(returns.iloc[:n], params).iloc[-1] if n >= MIN_RETURNS else np.nan
    return np.log(week_var)


class Garch(BaseEstimator, RegressorMixin):
    """GARCH(1,1) per company on daily close-to-close returns (Bollerslev, 1986), via `arch`.

    The forecast is the sum of the next five daily variance forecasts. Close-to-close variance
    includes the overnight move, which the Garman-Klass target leaves out, so a pooled line on
    the training rows maps the log forecast to the target's scale. Companies with fewer than
    `min_days` training returns use the median parameters of the others.

    returns: daily log close-to-close returns indexed by (ticker, date), sorted.
    """

    def __init__(  # noqa: D107
        self, returns: pd.Series, min_days: int = GARCH_MIN_DAYS, n_jobs: int = -1
    ) -> None:
        self.returns, self.min_days, self.n_jobs = returns, min_days, n_jobs

    def _series(self, X: pd.DataFrame) -> dict[str, pd.Series]:
        """Each company's percent returns up to the last day a row of X can use."""
        end = X["week"].max() + pd.Timedelta(days=7)
        r = self.returns[self.returns.index.get_level_values("date") <= end].dropna() * 100
        return {t: g.droplevel("ticker") for t, g in r.groupby(level="ticker")}

    def _log_forecast(self, X: pd.DataFrame) -> np.ndarray:
        series, out = self._series(X), pd.Series(np.nan, index=X.index)
        groups = [(t, w) for t, w in X.groupby("ticker")["week"]
                  if len(series.get(t, ())) >= MIN_RETURNS]  # fmt: skip
        jobs = (delayed(_garch_log_week_var)(series[t], self.params_.get(t, self.default_),
                                             pd.DatetimeIndex(w)) for t, w in groups)  # fmt: skip
        for (_, w), log_h in zip(groups, Parallel(n_jobs=self.n_jobs)(jobs), strict=True):
            out[w.index] = log_h
        return out.to_numpy()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Garch":  # noqa: D102
        series = self._series(X)
        eligible = [t for t in X["ticker"].unique() if len(series.get(t, ())) >= self.min_days]
        jobs = (delayed(_garch_params)(series[t].to_numpy()) for t in eligible)
        fits = Parallel(n_jobs=self.n_jobs)(jobs)
        if not fits:
            raise ValueError(f"no company has {self.min_days} daily returns in the training rows")
        self.params_ = dict(zip(eligible, fits, strict=True))
        self.default_ = np.median(np.vstack(fits), axis=0)
        log_h, y = self._log_forecast(X), np.asarray(y)
        ok = np.isfinite(log_h)
        self.calibration_ = LinearRegression().fit(log_h[ok, None], y[ok])
        self.fill_ = float(np.mean(y))  # rows with under MIN_RETURNS returns by their Friday
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: D102
        log_h = self._log_forecast(X)
        ok = np.isfinite(log_h)
        out = np.full(len(X), self.fill_)
        out[ok] = self.calibration_.predict(log_h[ok, None])
        return out
