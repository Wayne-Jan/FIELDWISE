"""Explanations of a FieldWise prediction, exact decompositions of the released linear components.

Field offset. FieldWise = Base + u. Correction only would add c = gamma G + kappa Cal; the decision layer replaces c by
u = (1 - alpha) u_SHRINK + alpha u_F16, where u_SHRINK scales c by clip(rhat / c, 0, 1) and u_F16 is the candidate
gamma' G + kappa' Cal closest to the readout rhat.

Environmental offset G. G = y_mean + sum_j (z_j - x_mean_j) beta_j over the standardised columns: named summaries and the
daily-weather lag features. A lag feature of variable v is F_vk = sum_l B_k(l) d_v(l) with d the standardised daily value
at lag l (days before the decision); its contribution splits exactly into one term per day,
    c_v(l) = d_v(l) w_v(l),  w_v(l) = sum_k B_k(l) beta_vk / sd_vk,
plus a constant. w_v(l) is the change in G per one-standard-deviation daily anomaly of v at lag l (the kernel).

Readouts. rhat_m = y_mean + z_B beta_B + z_F beta_F for each foundation model m; z_F splits into the decision-window
embedding, its change from the first decision window and its change from the pre-season window.
"""
import numpy as np
import pandas as pd

from .. import decision, environment
from .. import readout as R

DAILY_LABELS = ('mean temperature', 'max temperature', 'min temperature', 'max VPD', 'precipitation', 'solar radiation',
                'total evaporation', 'potential evaporation', 'root-zone soil water', 'deep soil water')


def field_offsets(bundle, cases):
    """Step-by-step field offset of every case (cases: pipeline.predict case table)."""
    k = bundle.config['strengths']
    G, Cal, rhat = (np.nan_to_num(cases[c].to_numpy(float)) for c in ('G', 'Cal', 'rhat'))
    c = k['gamma'] * G + k['kappa'] * Cal
    lam = np.clip(np.divide(rhat, c, out=np.zeros_like(c), where=c != 0), 0, 1)
    order = decision.candidate_order(np.full(len(c), k['gamma']), np.full(len(c), k['kappa']))
    U = G[:, None] * decision.F16[order, 0] + Cal[:, None] * decision.F16[order, 1]
    cost = U * U - 2 * U * rhat[:, None]
    j = (cost <= cost.min(-1, keepdims=True) + 1e-12).argmax(-1)
    pick = np.take_along_axis(order, j[:, None], 1)[:, 0]
    a = bundle.config['alpha']
    rh = cases[[f'rhat_{m}' for m in bundle.readouts]].to_numpy(float)
    out = cases[['case_id', 'fid', 'stage', 'base_mean']].copy()
    out['gamma_G'], out['kappa_Cal'], out['correction_only'] = k['gamma'] * G, k['kappa'] * Cal, c
    out['rhat'] = rhat
    out['readouts_agreeing'] = (np.sign(rh) == np.sign(rhat)[:, None]).sum(1)
    out['readout_same_direction_as_c'] = np.sign(rhat) == np.sign(c)
    out['shrink_lambda'], out['u_shrink'] = lam, lam * c
    out['f16_gamma'], out['f16_kappa'] = decision.F16[pick, 0], decision.F16[pick, 1]
    out['u_f16'] = np.take_along_axis(U, j[:, None], 1)[:, 0]
    out['alpha'] = a
    out['u'] = (1 - a) * out.u_shrink + a * out.u_f16
    out['fieldwise_mean'] = out.base_mean + out.u
    return out


def kernels(model):
    """[variables, lags] change in G per one-standard-deviation daily anomaly at each lag (lag 0 = last usable day)."""
    nv = len(model.daily_scaler[0])
    nb = environment.N_BASIS
    p = len(model.beta)
    seq = slice(p - nv * nb, p)
    mu, sd, has = (s[seq] for s in model.scaler)
    b = np.where(has, model.beta[seq] / sd, 0.0).reshape(nv, nb)
    return b @ environment.BASIS.T


def environment_contributions(model, named, lag):
    """Exact decomposition of G for every case.

    Named columns contribute (z_j - x_mean_j) beta_j (relative to their training mean). A daily variable contributes the
    sum of its per-day terms d_v(l) w_v(l), i.e. relative to days at the variable's historical mean (missing days count as
    average). The remaining constant, the same for every case, joins the intercept.
    Returns (named [cases, named columns], daily [cases, variables], per-day [cases, variables, lags], intercept [cases]),
    with G = intercept + named.sum(1) + daily.sum(1).
    """
    feat = environment.lag_features(lag, model.daily_scaler)
    X = np.column_stack([named, feat])
    mu, sd, has = model.scaler
    Z = np.nan_to_num((X - mu) / sd, nan=0.)
    Z[:, ~has] = 0.
    contrib = (Z - model.x_mean) * model.beta
    nn = named.shape[1]
    mu_d, sd_d = model.daily_scaler
    d = np.nan_to_num((lag - mu_d) / np.where(sd_d > 0, sd_d, 1))          # [cases, lags, variables]
    per_day = np.einsum('clv,vl->cvl', d, kernels(model))
    daily = per_day.sum(2)
    intercept = model.y_mean + contrib[:, nn:].sum(1) - daily.sum(1)
    return contrib[:, :nn], daily, per_day, intercept


def readout_contributions(bundle, stage, b7, means):
    """Per foundation model: intercept and the contributions of the Base-derived block and of the three embedding parts."""
    D = means.shape[-1]
    rows = []
    stage = np.asarray(stage)
    for j, fm in enumerate(bundle.readouts):
        full, first_only = R.window_blocks(means[:, :, j, :], np.full(len(b7), 2))
        for s in np.unique(stage):
            q = np.flatnonzero(stage == s)
            m = bundle.readouts[fm][s]
            first = s == bundle.config['stages'][0]
            XF = (first_only if first else full)[q]
            zb, zf = R._scale(b7[q], m.scalers[0]), R._scale(XF, m.scalers[1])
            pb = b7.shape[1]
            cb = zb @ m.beta[:pb]
            cf = zf * m.beta[pb:]
            parts = {'decision_window': cf[:, :D + 1].sum(1)}
            if first:
                parts['change_from_preseason'] = cf[:, D + 1:].sum(1)
                parts['change_from_first_decision'] = np.zeros(len(q))
            else:
                parts['change_from_first_decision'] = cf[:, D + 1:2 * D + 2].sum(1)
                parts['change_from_preseason'] = cf[:, 2 * D + 2:].sum(1)
            rows.append(pd.DataFrame(dict(row=q, model=fm, intercept=m.y_mean, base_features=cb, **parts)))
    out = pd.concat(rows).sort_values(['row', 'model']).reset_index(drop=True)
    out['rhat'] = out[['intercept', 'base_features', 'decision_window', 'change_from_first_decision', 'change_from_preseason']].sum(1)
    return out
