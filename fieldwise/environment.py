"""Environmental offset G: a two-block ridge readout of the field-mean residual from the growing-season environment.

Named block: decision-stage indicators, Base mean, Base standard deviation and log pixel count, seasonal exposure
summaries and land-surface temperature summaries. Sequence block: ten daily weather quantities over the 240 days before
the decision, each projected on eight cubic B-spline lag kernels. The two blocks have separate penalties, chosen per
target year by leave-one-year-out inside the legal history (years equal); the sequence penalty includes infinity, which
drops the block. With fewer than min_years legal years G is not fitted (missing, i.e. no correction).
"""
from dataclasses import dataclass

import numpy as np

from .linalg import apply_scaler, fit_scaler, ridge_two_block, standardize
from .schedule import DAY_MS, DECISION_DELAY_DAYS
from .selection import cutoffs, first_min, legal_mask, mse_after_shift, weighted_mean

LAMBDAS = (0.1, 1., 10., 100., 1e3, 1e4, 1e5)
SEQ_LAMBDAS = (1., 10., 100., 1e3, 1e4, 1e5, np.inf)
PAIRS = [(l1, l2) for l2 in SEQ_LAMBDAS[::-1] for l1 in LAMBDAS]
N_LAGS, N_BASIS = 240, 8


def bspline_basis(x, k, lo, hi, deg=3):
    """Uniform-knot B-spline basis (Cox-de Boor recursion): k functions on [lo, hi)."""
    inner = np.linspace(lo, hi, k - deg + 1)
    t = np.concatenate([[lo] * deg, inner, [hi] * deg])
    B = ((t[:-1][None] <= x[:, None]) & (x[:, None] < t[1:][None])).astype(float)
    for d in range(1, deg + 1):
        left = np.where(t[d:-1] > t[:-d - 1],
                        (x[:, None] - t[:-d - 1]) / np.where(t[d:-1] > t[:-d - 1], t[d:-1] - t[:-d - 1], 1), 0)
        right = np.where(t[d + 1:] > t[1:-d],
                         (t[d + 1:] - x[:, None]) / np.where(t[d + 1:] > t[1:-d], t[d + 1:] - t[1:-d], 1), 0)
        B = left * B[:, :-1] + right * B[:, 1:]
    return B


BASIS = bspline_basis(np.arange(N_LAGS, dtype=float), N_BASIS, 0., float(N_LAGS))


def lagged(daily, fs_row, n_days):
    """[cases, lags, variables]: lag 0 is the last usable day of each case; days before the sequence start are missing.

    daily: [field-seasons, days, variables]; fs_row: field-season of each case; n_days: usable days of each case.
    """
    idx = np.asarray(n_days)[:, None] - 1 - np.arange(N_LAGS)[None, :]
    out = daily[np.asarray(fs_row)[:, None], np.clip(idx, 0, None)]
    out[idx < 0] = np.nan
    return out


def day_available_ms(fs_first_day, n_days, delay_days=DECISION_DELAY_DAYS):
    """[field-seasons, days] time at which each daily value is available: the end of its UTC day plus the delay.

    fs_first_day: UTC day number (days since 1970-01-01) of the first row of each field-season's daily series.
    """
    return (np.asarray(fs_first_day)[:, None] + np.arange(n_days)[None, :] + 1 + delay_days) * DAY_MS


def daily_scaler(daily, fs_year, T, available_ms, cutoff_ms):
    """Mean and standard deviation of every daily variable over the field-seasons of years before T, using only the days
    available by the cutoff."""
    early = np.asarray(fs_year) < T
    D = np.where((available_ms > cutoff_ms)[:, :, None], np.nan, daily)[early].reshape(-1, daily.shape[-1])
    return np.nanmean(D, 0), np.nanstd(D, 0)


def lag_features(lag, scaler):
    """[cases, variables x basis]: standardised daily values (missing -> 0) projected on the lag kernels."""
    mu, sd = scaler
    Z = np.nan_to_num((lag - mu) / np.where(sd > 0, sd, 1))
    return np.einsum('clv,lk->cvk', Z, BASIS).reshape(len(lag), -1)


@dataclass
class YearModel:
    """G of one target year: apply to new cases with predict()."""
    year: int
    pair: tuple
    daily_scaler: tuple
    scaler: tuple
    x_mean: np.ndarray
    y_mean: float
    beta: np.ndarray

    def predict(self, named, lag):
        X = np.column_stack([named, lag_features(lag, self.daily_scaler)])
        return (apply_scaler(X, self.scaler) - self.x_mean) @ self.beta + self.y_mean


def fit_environment(cases, named, lag, daily, fs_year, fs_first_day, min_years=2, cutoff=None):
    """G for every case of every target year, with the fitted year models.

    cases: history and query cases, columns year, label_ms, decision_ms, w, rbar, mse (rbar, mse missing for cases without
    labels, e.g. the queries of a new year); only legal cases with a field-mean residual are fitted. named: [cases, p] named
    block. lag: lagged() of the cases. daily, fs_year, fs_first_day: the field-season daily array, years and first UTC day
    (the daily standardisation uses only days available by the cutoff). cutoff: year -> UTC ms (see selection.py).
    Returns (G [cases], {year: YearModel}, {year: selection record}).
    """
    year = cases.year.to_numpy()
    cut = cutoffs(cases, cutoff)
    avail = day_available_ms(fs_first_day, daily.shape[1])
    labelled = np.isfinite(cases.rbar.to_numpy(float))
    G = np.full(len(cases), np.nan)
    models, records = {}, {}
    for T in sorted(int(v) for v in np.unique(year)):
        legal = legal_mask(cases, T, cut) & labelled
        if not legal.any():
            continue
        dsc = daily_scaler(daily, fs_year, T, avail, cut[T])
        X = np.column_stack([named, lag_features(lag, dsc)])
        q = year == T
        L = cases[legal]
        XL, wL, yL = X[legal], L.w.to_numpy(), L.rbar.to_numpy()
        years = sorted(int(v) for v in L.year.unique())
        records[T] = dict(year=T, legal_years=years)
        if len(years) < min_years:
            continue
        scores = []
        for V in years:
            tr, va = (L.year != V).to_numpy(), (L.year == V).to_numpy()
            allV = year == V
            Ztr, ZV = standardize(XL[tr], wL[tr], X[allV])
            pV = ridge_two_block(Ztr, yL[tr], wL[tr], ZV, named.shape[1], PAIRS)[0]
            pv = pV[np.isin(cases.case_id[allV].to_numpy(), L.case_id[va].to_numpy())]
            Lv = L[va]
            scores.append(weighted_mean(Lv.w, mse_after_shift(Lv.rbar, Lv.mse, pv)))
        i = first_min(np.mean(scores, 0))
        scaler = fit_scaler(XL, wL)
        ZL, Zq = apply_scaler(XL, scaler), apply_scaler(X[q], scaler)
        pq, beta, xm, ym = ridge_two_block(ZL, yL, wL, Zq, named.shape[1], [PAIRS[i]], return_center=True)
        G[q] = pq[:, 0]
        models[T] = YearModel(T, PAIRS[i], dsc, scaler, xm, float(ym), beta[:, 0])
        records[T]['pair'] = [float(v) for v in PAIRS[i]]
    return G, models, records
