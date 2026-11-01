"""Strengths of the three corrections: prediction = Base + gamma G + kappa Cal + beta S.

gamma, kappa and beta take values in {0, 0.5, 1} (27 combinations), chosen once per target year by cumulative forward
selection over the earlier years and shared by its decision stages. Candidates are ordered by total strength so that near
ties keep the smaller intervention. Case losses are exact from per-case sums on labelled pixels.
"""
import itertools

import numpy as np

from .selection import forward_choice

GRID = (0.0, 0.5, 1.0)
COMBOS = np.array(sorted(itertools.product(GRID, GRID, GRID), key=lambda c: (sum(c), c)))   # (gamma, beta, kappa)

MOMENTS = ('n', 'sr', 'srr', 'sS', 'srS', 'sSS')


def shift_loss(m, u, beta):
    """Exact case MSE of Base + u + beta S on labelled pixels.

    m [..., 6]: n, sr, srr (residual sums), sS, srS, sSS (sums involving the within-field pattern S); u, beta broadcastable.
    """
    n, sr, srr, sS, srS, sSS = np.moveaxis(m, -1, 0)
    return (srr - 2 * u * sr - 2 * beta * srS + beta * beta * sSS + 2 * u * beta * sS) / n + u * u


def combination_losses(m, G, Cal):
    """[cases, 27] case MSE of every strength combination (missing G or Cal count as 0)."""
    g, b, k = (COMBOS[:, i][None, :] for i in range(3))
    c = g * np.nan_to_num(G)[:, None] + k * np.nan_to_num(Cal)[:, None]
    n, sr, srr, sS, srS, sSS = (m[:, j][:, None] for j in range(6))
    return (srr - 2 * b * srS + b ** 2 * sSS - 2 * c * (sr - b * sS) + n * c ** 2) / n


def select_strengths(cases, m, G, Cal, cutoff=None, targets=None):
    """{year: (gamma, beta, kappa)} by cumulative forward selection, and the case losses of all combinations.

    cases: labelled cases with year, w, label_ms, decision_ms; m: [cases, 6] moments in the same order. targets: years to
    choose for (default: the years of cases; a new year without labels needs its cutoff).
    """
    L = combination_losses(m, G, Cal)
    years = sorted(int(v) for v in cases.year.unique()) if targets is None else list(targets)
    pick = forward_choice(cases, L, np.ones(len(cases), bool), years, cutoff=cutoff)
    return {T: tuple(float(v) for v in COMBOS[j]) for T, (j, _) in pick.items()}, L
