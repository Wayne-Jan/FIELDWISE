"""Calibration offset K: the field-constant part of a linear recalibration of the Base.

For target year T and decision stage s, y = a + b * Base is fitted by weighted least squares on the labelled pixels of the
legal same-stage history (each case weighted by its field-season weight, shared by its labelled pixels). Only the field
constant is used: Cal = a + (b - 1) * mean Base prediction over the service pixels. No fit (Cal missing) with fewer than
min_years legal years or fewer than two history cases.
"""
import numpy as np
import pandas as pd

from .selection import cutoffs, legal_mask


def fit_affine(cases, min_years=2, cutoff=None, targets=None):
    """Per (year, stage) coefficients a, b.

    cases: labelled cases only (n > 0), columns year, stage, label_ms, decision_ms, w and the labelled-pixel means
    sy, sp, spp, syp (bank moments). targets: years to fit; by default the years of cases, each for the stages it has. Given
    targets (for example a new year without labels) are fitted for every stage of the history and need their cutoffs.
    """
    cut = cutoffs(cases, cutoff)
    rows = []
    for T in (sorted(cases.year.unique()) if targets is None else targets):
        L = cases[legal_mask(cases, T, cut)]
        if L.year.nunique() < min_years:
            continue
        for s in sorted(cases.stage.astype(str).unique()):
            h = L[L.stage == s]
            if len(h) < 2 or (targets is None and not ((cases.year == T) & (cases.stage == s)).any()):
                continue
            w = h.w / h.w.sum()
            Sp, Sy, Spp, Spy = (w * h.sp).sum(), (w * h.sy).sum(), (w * h.spp).sum(), (w * h.syp).sum()
            b = (Spy - Sp * Sy) / (Spp - Sp * Sp)
            rows.append(dict(year=int(T), stage=s, a=float(Sy - b * Sp), b=float(b), n_hist=len(h)))
    return pd.DataFrame(rows, columns=['year', 'stage', 'a', 'b', 'n_hist'])


def calibration_offset(coef, year, stage, p_mean_service):
    """Cal = a + (b - 1) * mean Base prediction over the service pixels; NaN where no coefficients exist."""
    ab = coef.set_index(['year', 'stage']).reindex(pd.MultiIndex.from_arrays([np.asarray(year), np.asarray(stage)]))
    return (ab.a + (ab.b - 1) * np.asarray(p_mean_service)).to_numpy()
