"""Legal history, field-season weights and forward selection rules shared by the correction components.

A case is one field-season at one decision time. Every target year T has a cutoff: the time by which the fits serving T
are made. In the manuscript it is D(T), the earliest decision of the phase (in-season decision times share one cutoff;
harvest time has its own). The legal history of T is every case of an earlier year whose decision and labels are both
available by the cutoff. Pass the cutoffs explicitly (cutoff: mapping year -> UTC ms); without them they are derived as
the earliest decision of each year among the cases given, which equals D(T) only when those cases include every query of
the phase and is undefined for a year without cases. Within a year each field-season counts once; its decision-time cases
share that weight.

Fit and serve one crop at a time: the cutoffs, history and references of a crop come from that crop only. The functions
take no crop argument; a table with a crop column holding several crops is rejected.
"""
import numpy as np
import pandas as pd

STRENGTHS = (0.0, 0.5, 1.0)
TIE = 1e-12
# decision stage -> phase; the stages of a phase share one cutoff per target year
PHASES = {'tau1': 'in_season', 'tau2': 'in_season', 'tau3': 'in_season', 'tau4': 'in_season', 'harvest': 'harvest'}


def one_crop(table):
    """Raise if table has a crop column with more than one crop."""
    if 'crop' in getattr(table, 'columns', ()) and table.crop.nunique() > 1:
        raise ValueError(f'one crop per call, got {sorted(table.crop.unique())}: fit each crop separately')


def field_season_weights(cases):
    """1 / number of cases of the same (year, field)."""
    key = cases.year.astype(str) + '_' + cases.field.astype(str)
    return (1.0 / key.map(key.value_counts())).to_numpy()


def earliest_decision(cases):
    """D(T) for every year T."""
    return cases.groupby('year').decision_ms.min()


def cutoff_table(queries, phases=PHASES):
    """Manuscript cutoffs: (year, phase, cutoff_ms) with cutoff = the earliest decision among every query of that year and
    phase. queries: all cases to be served (labelled or not), columns year, stage, decision_ms."""
    one_crop(queries)
    q = queries.assign(phase=queries.stage.map(phases))
    if q.phase.isna().any():
        raise ValueError(f'stages without a phase: {sorted(q.stage[q.phase.isna()].unique())}')
    return q.groupby(['year', 'phase']).decision_ms.min().rename('cutoff_ms').reset_index()


def phase_cutoffs(table, phase):
    """{year: cutoff_ms} of one phase, the cutoff argument of the fitting functions."""
    one_crop(table)
    t = table[table.phase == phase]
    return dict(zip(t.year.astype(int), t.cutoff_ms.astype(float)))


def cutoffs(cases, cutoff=None):
    """Cutoff of every target year: the given mapping (year -> UTC ms) or the earliest decision of each year in cases."""
    one_crop(cases)
    return earliest_decision(cases) if cutoff is None else pd.Series(cutoff, dtype='float64')


def legal_mask(cases, T, cutoff=None):
    """Cases of years before T whose decision and labels are both available by the cutoff of T."""
    c = cutoffs(cases, cutoff)[T]
    return ((cases.year < T) & (cases.label_ms <= c) & (cases.decision_ms <= c)).to_numpy()


def mse_after_shift(rbar, mse, c):
    """Case MSE on labelled pixels after adding a field constant c ([cases] or [cases, k]): mse - 2 c rbar + c^2."""
    c = np.asarray(c, float)
    rbar, mse = np.asarray(rbar, float), np.asarray(mse, float)
    if c.ndim == 2:
        rbar, mse = rbar[:, None], mse[:, None]
    return mse - 2 * c * rbar + c ** 2


def weighted_mean(w, A):
    """Weighted mean over cases of A [cases] or [cases, k]."""
    w = np.asarray(w, float)
    return (w[:, None] * A).sum(0) / w.sum() if np.ndim(A) == 2 else float((w * A).sum() / w.sum())


def first_min(scores):
    """Index of the first candidate within TIE of the minimum (candidates are ordered simplest first)."""
    s = np.asarray(scores, float)
    return int(np.argmax(s <= s.min() + TIE))


def leave_one_year_out(L, fit_predict, n_candidates):
    """Pick a hyperparameter by leave-one-year-out inside the legal history L (years weighted equally).

    fit_predict(train, valid) returns field constants [valid cases, n_candidates]. L needs columns year, w, rbar, mse.
    Returns (index, scores, {year: fold predictions}).
    """
    scores, folds = [], {}
    for V in sorted(L.year.unique()):
        tr, va = L[L.year != V], L[L.year == V]
        P = fit_predict(tr, va)
        folds[V] = P
        scores.append(weighted_mean(va.w, mse_after_shift(va.rbar, va.mse, P)))
    scores = np.mean(scores, 0)
    return first_min(scores), scores, folds


def forward_choice(cases, losses, has, targets, allowed=None, cutoff=None):
    """Cumulative forward selection of one candidate per target year.

    losses [cases, k]: case loss of each candidate (index 0 = no correction). For target year T the score of a candidate is
    the year-equal weighted mean loss over the earlier years V < T, on the legal cases of V that have their own rolling
    prediction (has). No such year -> candidate 0.
    Returns {T: (index, development years)}.
    """
    cut = cutoffs(cases, cutoff)
    out = {}
    for T in targets:
        ok = has & legal_mask(cases, T, cut)
        d = cases[ok]
        devs = sorted(int(v) for v in d.year.unique())
        if not devs:
            out[T] = (0, devs)
            continue
        Ld = losses[ok]
        sc = np.mean([weighted_mean(d.w[d.year == V], Ld[(d.year == V).to_numpy()]) for V in devs], 0)
        if allowed is not None:
            sc = np.where(allowed, sc, np.inf)
        out[T] = (first_min(sc), devs)
    return out


def year_table(R, cols):
    """Year-by-year weighted means of the case-loss columns."""
    return R.groupby('year').apply(lambda d: pd.Series({k: weighted_mean(d.w, d[k].to_numpy()) for k in cols}))
