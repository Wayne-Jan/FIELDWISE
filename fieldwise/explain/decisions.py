"""Decision comparisons and failure diagnostics (manuscript Sections 4.7.2 and 4.8, Appendix D.3).

Fixed fractions. Base + beta S15 + lambda c applies the same fraction lambda of the Correction-only offset c to every case
(lambda = 0 keeps RPT only, lambda = 1 is Correction only). Case MSE is quadratic in lambda and exact from per-case sums.
The historical fallback picks lambda_T from {0, 0.25, 0.5, 0.75, 1} by the earlier evaluation years.

Readout permutation. FM4 readouts are exchanged among the cases of a group (for example year x decision time, or country x
year x decision time); each recipient recomputes its offset from the assigned readout with its own G, K, strengths and
alpha. Losses and severe-harm indicators are averaged over the permutations.

Loss decomposition. A field-constant adjustment x acting on a mean residual e changes case MSE by x^2 - 2 x e. Adjustments
with |x| <= 1e-9 are none; x e < 0 is the wrong direction; otherwise x / e > 2 overshoots (the change is positive) and the
rest is beneficial.

Readout diagnostics. Averaging gain = mean single-readout MSE - MSE of their mean = mean readout spread (exact); the
same-sign error share counts cases whose readout errors are all strictly positive or all strictly negative. Readout skill
against the field-mean residual: MSE, predictive R^2, bias, signed correlation, calibration slope, and the covariance split
between year x decision-time groups and within them.
"""
import numpy as np
import pandas as pd

from ..decision import f16, shrink
from ..evaluation import SEVERE, scalar_rollup
from ..selection import cutoffs, legal_mask
from ..strength import shift_loss

FRACTIONS = np.array([0., .25, .5, .75, 1.])
CATEGORIES = ('none', 'wrong_direction', 'overshoot', 'beneficial')


def fraction_losses(m, c, beta, lam=FRACTIONS):
    """[cases, fractions] exact case MSE of Base + beta S15 + lam c; m: strength.MOMENTS sums."""
    lam = np.asarray(lam, float)
    return shift_loss(m[:, None, :], np.asarray(c, float)[:, None] * lam[None, :], np.asarray(beta, float)[:, None])


def fallback_fraction(cases, losses, evaluated, cutoff=None):
    """{year: index into FRACTIONS} for every evaluated target year.

    The score of a fraction is the year-equal weighted MSE over the legal earlier evaluation-year cases (decision and
    labels available by the target year's cutoff, selection.py); the first fraction within 1e-12 of the lowest wins; no
    history -> lambda = 1. cases: year, w, label_ms, decision_ms; losses: fraction_losses().
    """
    yr, w = cases.year.to_numpy(), cases.w.to_numpy(np.float64)
    cut = cutoffs(cases, cutoff)
    out = {}
    for T in np.unique(yr[evaluated]):
        prior = evaluated & legal_mask(cases, T, cut)
        if not prior.any():
            out[int(T)] = losses.shape[1] - 1
            continue
        s = scalar_rollup(yr[prior], w[prior], losses[prior])
        out[int(T)] = int(np.flatnonzero(s <= s.min() + 1e-12)[0])
    return out


def layer_offsets(rhat, G, Cal, c, order, alpha, has):
    """Shrinkage and FieldWise offsets for readouts rhat [..., cases] (a leading axis, e.g. permutations, broadcasts).

    G, Cal, c, alpha, has (readout available) per case; order: decision.candidate_order(); cases without a readout keep c.
    """
    r0 = np.where(has, rhat, 0.)
    uS = shrink(r0, c)
    uF = f16(r0, np.nan_to_num(G), np.nan_to_num(Cal), order)
    return np.where(has, uS, c), np.where(has, (1 - alpha) * uS + alpha * uF, c)


def permutation_index(n, groups, rng, n_perm=200):
    """[n_perm, n] donor index of every case: cases exchange readouts within each group.

    groups: list of case-index arrays (each sorted by case id, the list in sorted group-key order); cases outside the groups
    keep their own readout. One rng.permutation call per group and permutation, in this order.
    """
    idx = np.repeat(np.arange(n)[None, :], n_perm, 0)
    for b in range(n_perm):
        for ix in groups:
            idx[b, ix] = rng.permutation(ix)
    return idx


def permuted_readouts(rhat, groups, rng, n_perm=200):
    """[n_perm, cases] readouts permuted within each group (permutation_index)."""
    return np.asarray(rhat, float)[permutation_index(len(rhat), groups, rng, n_perm)]


def permutation_groups(cases, mask, keys):
    """Case-index groups of the cases in mask, by keys (sorted), each sorted by case_id."""
    d = cases[mask]
    return [np.array(sorted(ix, key=lambda i: cases.case_id.iloc[i])) for _, ix in d.groupby(keys).groups.items()]


def permutation_losses(R, m, beta, base, decide):
    """Mean case loss and severe-harm share over permutations for each policy returned by decide(R) (tuple of [perms,
    cases] offsets). Returns ([cases, policies], [cases, policies])."""
    L, H = [], []
    for u in decide(R):
        loss = shift_loss(m[None, :, :], u, beta[None, :])
        L.append(loss.mean(0))
        H.append((loss > SEVERE * base[None, :] + 1e-12).mean(0))
    return np.column_stack(L), np.column_stack(H)


def loss_change(x, e):
    """(change in case MSE x^2 - 2 x e, {category: mask}) for adjustments x on mean residuals e."""
    x, e = np.asarray(x, float), np.asarray(e, float)
    none = np.abs(x) <= 1e-9
    ratio = np.divide(x, e, out=np.full_like(x, np.inf), where=e != 0)
    wrong = ~none & (x * e < 0)
    over = ~none & ~wrong & (ratio > 2)
    return x * x - 2 * x * e, dict(none=none, wrong_direction=wrong, overshoot=over, beneficial=~none & ~wrong & ~over)


def normalized_weights(W, col, year, w):
    """[draws, cases] weights summing to 1 per draw: draw counts x case weights, normalised within each year, years equal
    among the years present in the draw."""
    raw = W[:, col].astype(np.float64) * np.asarray(w, float)[None]
    yi, years = pd.factorize(np.asarray(year), sort=True)
    den = raw @ np.eye(len(years))[yi]
    out = np.divide(raw, den[:, yi], out=np.zeros_like(raw), where=den[:, yi] > 0)
    return out / (den > 0).sum(1)[:, None]


def _div(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 1e-14)


def readout_skill(P, X, y, groups):
    """Skill of readouts X [cases, k] for field-mean residuals y under weights P [draws, cases].

    groups: case-index arrays (year x decision time) for the between/within covariance split. Returns two dicts of
    [draws, k]: mse, r2, bias, signed_r, slope; and cov_total, cov_between, cov_within, pred_var_between, pred_var_within,
    signed_r_between, signed_r_within, slope_between, slope_within.
    """
    k = X.shape[1]
    A = P @ np.column_stack([X, X * X, X * y[:, None], y, y * y, (X - y[:, None]) ** 2])
    mx, x2, xy = A[:, :k], A[:, k:2 * k], A[:, 2 * k:3 * k]
    my, y2 = A[:, [3 * k]], A[:, [3 * k + 1]]
    mse = A[:, 3 * k + 2:]
    vx, vy = np.maximum(x2 - mx * mx, 0), np.maximum(y2 - my * my, 0)
    cov = xy - mx * my
    bxy, bx2, by2 = np.zeros_like(cov), np.zeros_like(cov), np.zeros_like(my)
    for ix in groups:
        Pg = P[:, ix]
        sg = Pg.sum(1)[:, None]
        sx, sy = Pg @ X[ix], Pg @ y[ix, None]
        bxy += np.divide(sx * sy, sg, out=np.zeros_like(sx), where=sg > 0)
        bx2 += np.divide(sx * sx, sg, out=np.zeros_like(sx), where=sg > 0)
        by2 += np.divide(sy * sy, sg, out=np.zeros_like(sy), where=sg > 0)
    cb, cw = bxy - mx * my, xy - bxy
    vxb, vyb = np.maximum(bx2 - mx * mx, 0), np.maximum(by2 - my * my, 0)
    vxw, vyw = np.maximum(x2 - bx2, 0), np.maximum(y2 - by2, 0)
    main = dict(mse=mse, r2=1 - _div(mse, vy), bias=mx - my, signed_r=_div(cov, np.sqrt(vx * vy)), slope=_div(cov, vx))
    scale = dict(cov_total=cov, cov_between=cb, cov_within=cw, pred_var_between=vxb, pred_var_within=vxw,
                 signed_r_between=_div(cb, np.sqrt(vxb * vyb)), signed_r_within=_div(cw, np.sqrt(vxw * vyw)),
                 slope_between=_div(cb, vxb), slope_within=_div(cw, vxw))
    return main, scale


def readout_agreement(P, X4, y):
    """Averaging diagnostics of single readouts X4 [cases, m] and their mean under weights P [draws, cases]:
    mean single MSE, MSE of the mean, averaging gain (= weighted mean spread) and same-sign error share (%)."""
    E = X4 - y[:, None]
    single = (P @ (E * E)).mean(1)
    mean_mse = P @ (X4.mean(1) - y) ** 2
    same = ((E > 0).all(1) | (E < 0).all(1)).astype(float)
    return dict(mean_single_mse=single, ensemble_mse=mean_mse, diversity_gain=single - mean_mse, same_sign_error_pct=100 * (P @ same))
