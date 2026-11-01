"""Field-mean residual readouts from foundation-model embeddings (two-block ridge).

For each decision stage and target year, rbar is regressed on two blocks fitted on the legal history:
  B: seven Base-derived features (Base mean, Base standard deviation, log pixel count, G, Cal and missing flags of G, Cal);
  F: the field-mean embedding of the decision window (M) and its changes from the first decision window and from the
     pre-season window (D), each with a missing-window flag.
Columns are z-scored on the training rows; the penalty pair is chosen by leave-one-year-out (years equal, unweighted
rows) and refitted on all training rows. The readout of several foundation models is the mean of their predictions.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .selection import cutoffs, legal_mask

LAM_B = np.array([0.1, 1., 10., 100., 1000.])
LAM_F = np.array([1., 10., 100., 1e3, 1e4, 1e5])
COMBOS = np.array([(a, b) for a in LAM_B for b in LAM_F])


def field_means(emb, starts):
    """Field-mean embeddings of consecutive pixel blocks starting at starts: float32 rows, rows with a missing value are
    left out, float64 sums, float32 means (NaN for a block without a valid row). Returns (means, valid rows per block)."""
    X = np.asarray(emb, dtype=np.float32)
    valid = np.isfinite(X).all(1)
    X[~valid] = 0
    S = np.add.reduceat(X, starts, axis=0, dtype=np.float64)
    N = np.add.reduceat(valid.astype(np.int64), starts)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(N[:, None] > 0, S / np.maximum(N, 1)[:, None], np.nan).astype('float32'), N


def window_blocks(means, dslot):
    """Feature block F for every case.

    means: [cases, slots, D] field-mean embeddings by window (slot 0 = pre-season, 1 = first decision, ...); NaN where a
    window is absent. dslot: decision slot of each case. Returns (F, first) where cases at the first decision use only the
    change from the pre-season window (first marks them).
    """
    n, _, D = means.shape
    ci = np.arange(n)
    miss = ~np.isfinite(means).all(2)
    md, m1, m0 = miss[ci, dslot], miss[:, 1], miss[:, 0]
    M = np.hstack([means[ci, dslot], md[:, None].astype(float)])
    from_first = np.hstack([M[:, :D] - means[:, 1], (md | m1)[:, None].astype(float)])
    from_pre = np.hstack([M[:, :D] - means[:, 0], (md | m0)[:, None].astype(float)])
    full = np.hstack([M, from_first, from_pre])
    first_only = np.hstack([M, from_pre])
    return full, first_only


def _scaler(X, fit):
    mu, sd = np.nanmean(X[fit], 0), np.nanstd(X[fit], 0)
    keep = np.isfinite(sd) & (sd > 1e-8)
    return np.nan_to_num(mu), sd, keep


def _scale(X, scaler):
    mu, sd, keep = scaler
    return np.nan_to_num(np.where(keep, (X - mu) / np.where(keep, sd, 1.), 0.), nan=0.)


def _solve(XB, XF, y, fit, combos):
    """Two-block ridge for every penalty pair: (scalers, y mean, coefficients [pairs, p])."""
    sc = (_scaler(XB, fit), _scaler(XF, fit))
    Z = np.hstack([_scale(XB, sc[0]), _scale(XF, sc[1])])
    Zf = Z[fit]
    ym = y[fit].mean()
    G, b = Zf.T @ Zf, Zf.T @ (y[fit] - ym)
    pen = np.where(np.arange(Z.shape[1])[None] < XB.shape[1], combos[:, :1], combos[:, 1:])
    beta = np.linalg.solve(G[None] + pen[:, :, None] * np.eye(Z.shape[1])[None],
                           np.broadcast_to(b, (len(combos), len(b)))[..., None])[..., 0]
    return sc, ym, beta, Z


@dataclass
class ReadoutModel:
    scalers: tuple
    y_mean: float
    beta: np.ndarray
    lambdas: tuple

    def predict(self, XB, XF):
        Z = np.hstack([_scale(XB, self.scalers[0]), _scale(XF, self.scalers[1])])
        return self.y_mean + Z @ self.beta


def fit_readout(XB, XF, y, year, train):
    """Penalty pair by leave-one-year-out within the training rows, then the refit on all training rows."""
    losses = []
    for v in np.unique(year[train]):
        fit, va = train[year[train] != v], train[year[train] == v]
        _, ym, beta, Z = _solve(XB, XF, y, fit, COMBOS)
        losses.append(((ym + beta @ Z[va].T - y[va][None]) ** 2).mean(1))
    k = int(np.argmin(np.mean(losses, 0)))
    sc, ym, beta, _ = _solve(XB, XF, y, train, COMBOS[k:k + 1])
    return ReadoutModel(sc, float(ym), beta[0], tuple(COMBOS[k]))


def legal_origins(cases, min_years=2, cutoff=None, labelled=None):
    """Target years with at least min_years legal years, and the legal training mask of each [cases, origins].

    A training case for target year T is from an earlier year, with its labels available and its decision made by the
    cutoff of T (selection.py), and labelled (a finite field-mean residual) when labelled is given.
    """
    year = cases.year.to_numpy()
    cut = cutoffs(cases, cutoff)
    lab_ok = np.ones(len(cases), bool) if labelled is None else np.asarray(labelled, bool)
    origins, cols = [], []
    for T in sorted(int(v) for v in np.unique(year)):
        m = legal_mask(cases, T, cut) & lab_ok
        if len(np.unique(year[m])) >= min_years:
            origins.append(int(T))
            cols.append(m)
    return np.array(origins), (np.column_stack(cols) if cols else np.zeros((len(cases), 0), bool))


def readouts(cases, b7, means, dslot, rbar, cutoff=None):
    """rhat of every case of every target year with a legal history, for one foundation model.

    cases: columns year, stage, label_ms, decision_ms; rbar missing for cases without labels (e.g. a new year's queries).
    cutoff: year -> UTC ms (selection.py). Returns (rhat [cases], {(stage, year): ReadoutModel}).
    """
    full, first_only = window_blocks(means, dslot)
    first_stage = sorted(cases.stage.unique())[0]
    year, stage = cases.year.to_numpy(), cases.stage.to_numpy()
    origins, legal = legal_origins(cases, cutoff=cutoff, labelled=np.isfinite(np.asarray(rbar, float)))
    rhat = np.full(len(cases), np.nan)
    models = {}
    for t in sorted(np.unique(stage)):
        XF = first_only if t == first_stage else full
        g = stage == t
        for oi, T in enumerate(origins):
            train = np.flatnonzero(g & legal[:, oi])
            query = np.flatnonzero(g & (year == T))
            if len(np.unique(year[train])) < 2 or not len(query):
                continue
            m = fit_readout(b7, XF, rbar, year, train)
            rhat[query] = m.predict(b7[query], XF[query])
            models[(t, int(T))] = m
    return rhat, models


def b7_features(p_mean, p_sd, log_n, G, Cal):
    """The seven Base-derived readout features: Base mean, standard deviation, log pixel count, G, Cal (missing -> 0), flags."""
    return np.column_stack([p_mean, p_sd, log_n, np.nan_to_num(G), np.nan_to_num(Cal), np.isnan(G).astype(float),
                            np.isnan(Cal).astype(float)])
