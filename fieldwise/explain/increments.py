"""Predictive increments of field-mean readouts (manuscript Section 4.7.1, Appendix D.1).

A readout regresses the Base field-mean residual on blocks of features: the seven shared features (readout.py), optionally
one FM block (readout.window_blocks) and optionally one named block (for example vegetation indices, LAI/FPAR, sowing
timing, soil and terrain). Every block is z-scored on the fitting rows and has its own ridge penalty; the penalties are
chosen jointly by leave-one-year-out within the training years (years equal, rows equal) and the model is refitted on all
training rows. Exact ties keep the first combination: smallest shared penalty, then FM, then named.

With L the year-equal weighted MSE of rbar - rhat over the evaluation cases and M_Base the Base total MSE (% units):
  I_X = 100 (L_S - L_X) / M_Base for X in {F, N, FN} (S: shared features only);
  I_F|N = I_FN - I_N, I_N|F = I_FN - I_F, shared = I_F + I_N - I_FN.
FM4 averages four separately fitted readouts (each with the same named block for the joint readout).
"""
import itertools

import numpy as np

from ..readout import LAM_B, LAM_F, _scale, _scaler

CHUNK = 30


def solve_blocks(blocks, y, fit, pred, combos):
    """Ridge predictions [combinations, pred rows] for every penalty combination (one column per non-empty block)."""
    blocks = [X for X in blocks if X.shape[1]]
    Z = np.hstack([_scale(X, _scaler(X, fit)) for X in blocks])
    Zf, Zp = Z[fit], Z[pred]
    ym = y[fit].mean()
    G, b = Zf.T @ Zf, Zf.T @ (y[fit] - ym)
    blk = np.concatenate([np.full(X.shape[1], i) for i, X in enumerate(blocks)])
    out = []
    for c in range(0, len(combos), CHUNK):
        pen = combos[c:c + CHUNK][:, blk]
        A = G[None] + pen[:, :, None] * np.eye(Z.shape[1])[None]
        beta = np.linalg.solve(A, np.broadcast_to(b, (len(pen), len(b)))[..., None])[..., 0]
        out.append(ym + beta @ Zp.T)
    return np.vstack(out)


def fit_blocks(blocks, y, year, train, query, grids=None):
    """Readout of the query rows from the training rows: (rhat [query], penalties).

    blocks: list of [cases, p] arrays (the first is the shared block; empty blocks are skipped); grids: penalty grid of
    each non-empty block (default: LAM_B for the first, LAM_F for the others).
    """
    used = [X for X in blocks if X.shape[1]]
    grids = grids or [LAM_B] + [LAM_F] * (len(used) - 1)
    combos = np.array(list(itertools.product(*grids)))
    losses = []
    for v in np.unique(year[train]):
        fit, va = train[year[train] != v], train[year[train] == v]
        losses.append(((solve_blocks(used, y, fit, va, combos) - y[va][None]) ** 2).mean(1))
    k = int(np.argmin(np.mean(losses, 0)))
    return solve_blocks(used, y, train, query, combos[k:k + 1])[0], combos[k]


def year_mean(W, col, year, w, V):
    """[draws, columns] year-equal weighted means of V [cases, columns] for every draw row of W (evaluation.draw_matrix;
    col: W column of each case)."""
    Wc = W[:, col].astype(np.float64) * np.asarray(w, float)[None]
    yi = np.unique(year, return_inverse=True)[1]
    Gy = np.eye(yi.max() + 1)[yi]
    den = Wc @ Gy
    return np.stack([((Wc * V[:, j]) @ Gy / den).mean(1) for j in range(V.shape[1])], 1)


def readout_losses(W, col, year, w, rbar, R):
    """[draws, readouts] year-equal weighted MSE of rbar - R[:, j]."""
    return year_mean(W, col, year, w, (np.asarray(rbar, float)[:, None] - R) ** 2)


def increments(L_S, L_F, L_N, L_FN, M_base):
    """Increments and their split (% of Base total MSE); arrays per bootstrap draw."""
    I_F, I_N, I_FN = (100 * (L_S - L) / M_base for L in (L_F, L_N, L_FN))
    return dict(I_F=I_F, I_N=I_N, I_FN=I_FN, F_given_N=I_FN - I_N, N_given_F=I_FN - I_F, shared=I_F + I_N - I_FN)
