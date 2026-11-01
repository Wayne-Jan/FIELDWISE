"""Decision layer: how much of the field-mean offset to apply to each field, guided by its residual readout.

Correction only applies c = gamma G + kappa Cal. With a readout rhat the layer chooses a field constant u:
  SHRINK: u = lambda c with lambda = clip(rhat / c, 0, 1) (0 when c = 0);
  F16:    u = the candidate gamma' G + kappa' Cal, gamma', kappa' in {0, 0.25, 0.5, 1}, closest to rhat (minimises
          u^2 - 2 u rhat; ties keep the current (gamma, kappa), then the smaller total);
  mix:    u = (1 - alpha) u_SHRINK + alpha u_F16.
alpha in {0, 0.1, ..., 1} is chosen per target year from the earlier evaluation years: the lowest year-equal loss among the
alphas whose severe-harm share there is not above Correction only (none -> 0; no earlier year -> 0). A field without a
readout keeps c. The prediction is Base + beta S15 + u.
"""
import numpy as np
import pandas as pd

from .evaluation import SEVERE, scalar_rollup
from .selection import cutoffs, legal_mask
from .strength import shift_loss

GRID = np.array([0., .25, .5, 1.])
F16 = np.array([(a, b) for a in GRID for b in GRID])
ALPHAS = np.round(np.arange(11) / 10, 1)


def f16_order(gamma, kappa):
    """Candidate order for ties: the current (gamma, kappa) first, then by total strength and gamma."""
    first = int(np.flatnonzero((F16[:, 0] == gamma) & (F16[:, 1] == kappa))[0])
    rest = sorted((i for i in range(16) if i != first), key=lambda i: (F16[i].sum(), F16[i, 0]))
    return np.array([first, *rest])


def candidate_order(gamma, kappa):
    """[cases, 16] tie order for per-case strengths."""
    order = np.zeros((len(gamma), 16), int)
    for (g, k), ix in pd.DataFrame(dict(g=gamma, k=kappa)).groupby(['g', 'k']).indices.items():
        order[ix] = f16_order(g, k)
    return order


def shrink(rhat, c):
    lam = np.clip(np.divide(rhat, c, out=np.zeros(np.broadcast_shapes(rhat.shape, c.shape)), where=c != 0), 0., 1.)
    return lam * c


def f16(rhat, G, Cal, order):
    U = G[:, None] * F16[order, 0] + Cal[:, None] * F16[order, 1]
    cost = U * U - 2 * U * rhat[..., None]
    k = (cost <= cost.min(-1, keepdims=True) + 1e-12).argmax(-1)
    return np.take_along_axis(np.broadcast_to(U, cost.shape), k[..., None], -1)[..., 0]


def offsets(rhat, G, Cal, gamma, kappa):
    """u of every alpha [cases, alphas] and the Correction-only offset c; G, Cal missing count as 0, rhat missing -> c."""
    G, Cal = np.nan_to_num(G), np.nan_to_num(Cal)
    c = gamma * G + kappa * Cal
    ok = np.isfinite(rhat)
    r0 = np.where(ok, rhat, 0.)
    uS, uF = shrink(r0, c), f16(r0, G, Cal, candidate_order(gamma, kappa))
    U = np.column_stack([(1 - a) * uS + a * uF for a in ALPHAS])
    return np.where(ok[:, None], U, c[:, None]), c


def alpha_losses(m, U, c, beta):
    """[cases, 2 + alphas]: Base, Correction only and every alpha (exact case MSE)."""
    base = m[:, 2] / m[:, 0]
    LB = shift_loss(m, c, beta)
    return np.column_stack([base, LB] + [shift_loss(m, U[:, j], beta) for j in range(U.shape[1])])


def choose_alpha(cases, losses, evaluated, cutoff=None, targets=None):
    """{year: alpha index} for every target year (the harm-constrained forward rule).

    cases: columns year, w, label_ms, decision_ms; losses: alpha_losses(); evaluated: boolean mask of evaluation cases.
    cutoff: year -> UTC ms (selection.py); targets: years to choose for (default: the evaluated years; a new year without
    labels needs its cutoff).
    """
    yr, w = cases.year.to_numpy(), cases.w.to_numpy(np.float64)
    cut = cutoffs(cases, cutoff)
    out = {}
    for T in (np.unique(yr[evaluated]) if targets is None else targets):
        prior = evaluated & legal_mask(cases, T, cut)
        j = 0
        if prior.any():
            Lp = losses[prior]
            sc = scalar_rollup(yr[prior], w[prior], Lp)
            hz = scalar_rollup(yr[prior], w[prior], (Lp > Lp[:, [0]] * SEVERE + 1e-12).astype(float))
            idx = sc[2:] / sc[0]
            feasible = np.flatnonzero(hz[2:] <= hz[1] + 1e-12)
            j = int(feasible[np.argmin(idx[feasible])]) if len(feasible) else 0
        out[int(T)] = j
    return out


def predict(base_pixels, S15_pixels, beta, u):
    """FieldWise prediction of one case: Base + beta S15 + u."""
    return base_pixels + beta * S15_pixels + u
