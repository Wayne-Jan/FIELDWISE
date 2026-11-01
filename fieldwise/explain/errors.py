"""Where the error sits and which correction removes it (manuscript Sections 4.5.1-4.5.2, Appendix C.1).

Layers. With e = y - prediction over the labelled pixels of a case, the field-mean residual is x = mean(e) and the
within-field variance v = mean((e - x)^2), so case MSE = x^2 + v. The field-mean residual splits into the weighted mean of
its year x decision-time group, successive deviations of nested groups (for example hemisphere, country, sowing group) and
the field-specific remainder. When the group means and the aggregation use the same weights, the cross terms vanish and
the weighted squared terms add up to the weighted MSE in every year.

Seed averaging. Per pixel, mean_s (y - p_s)^2 = (y - mean_s p_s)^2 + mean_s (p_s - mean_s p_s)^2, so the mean single-seed
MSE exceeds the ensemble MSE by the seed variance after any weighted aggregation, and layer by layer.

Contributions. For components C (RPT, G, K) at fixed strengths, the Shapley contribution of j is the mean over the orders
of adding components of the reduction L(A) - L(A + j); the contributions add up to L(Base) - L(Correction only) in every
layer and every bootstrap draw.

Years are weighted equally (as evaluation.py). Bootstrap draws enter as rows of a count matrix (evaluation.draw_matrix), so
everything is a set of sparse products with no loop over draws.
"""
import itertools
from math import factorial

import numpy as np
import pandas as pd
from scipy import sparse



def _onehot(codes, n):
    return sparse.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes), n))


def _keys(year, stage, groups):
    """Cumulative group keys: year x stage, then each nested grouping added in turn."""
    key = pd.Series(year).astype(str).to_numpy(object) + '_' + pd.Series(stage).astype(str).to_numpy(object)
    out = [key]
    for g in groups:
        key = key + '_' + pd.Series(g).astype(str).to_numpy(object)
        out.append(key)
    return out


def _design(keys, yi, ny):
    """Per key: case -> group and group -> year one-hot matrices; and case -> year."""
    d = []
    for k in keys:
        g = pd.factorize(k)[0]
        ng = g.max() + 1
        gy = np.zeros(ng, int)
        gy[g] = yi                                                              # a group lies inside one year
        d.append((_onehot(g, ng), _onehot(gy, ny)))
    return d, _onehot(yi, ny)


def year_layers(Wc, x, mse, design, Gy):
    """[draws, years, levels + 2] per-year layer values for every weight row of Wc [draws, cases]: the first group level,
    the deviations of each further level, the field-specific deviation and the within-field variance, each divided by the
    year's total weight."""
    s0y = np.asarray((Gy.T @ Wc.T).T)
    Q = []
    for G, Y in design:
        s0, s1 = np.asarray((G.T @ Wc.T).T), np.asarray((G.T @ (Wc * x).T).T)
        Q.append(np.asarray((Y.T @ np.where(s0 > 0, s1 ** 2 / np.where(s0 > 0, s0, 1.), 0.).T).T))
    Q.append(np.asarray((Gy.T @ (Wc * x ** 2).T).T))
    Wt = np.asarray((Gy.T @ (Wc * mse).T).T)
    den = np.where(s0y > 0, s0y, np.nan)
    return np.stack([Q[0] / den] + [(Q[i + 1] - Q[i]) / den for i in range(len(design))] + [(Wt - Q[-1]) / den], 2)


class Layers:
    """Layer decomposition of one set of cases.

    year, stage: per case; w: case weights (evaluation weights: field-seasons equal within a year, split over their cases);
    groups: {name: per-case label}, each grouping nested in the previous one inside year x stage; fid, W: field-season of
    every case and a draw matrix from evaluation.draw_matrix (fids, W); without them a single row with every case once.
    """

    def __init__(self, year, stage, w, groups=None, fid=None, draws=None):
        groups = dict(groups or {})
        self.names = ['year_stage'] + list(groups) + ['field', 'within']
        self.w = np.asarray(w, float)
        ys, self.yi = np.unique(np.asarray(year), return_inverse=True)
        self.design, self.Gy = _design(_keys(year, stage, groups.values()), self.yi, len(ys))
        if draws is None:
            self.Wc = self.w[None, :]
        else:
            fids, W = draws
            col = np.searchsorted(fids, np.asarray(fid))
            assert np.array_equal(fids[col], np.asarray(fid)), 'a case field-season is missing from the draw matrix'
            self.Wc = W[:, col].astype(np.float64) * self.w[None, :]

    def per_year(self, x, mse):
        """[draws, years, layers] for field-mean residuals x and case MSE."""
        return year_layers(self.Wc, np.asarray(x, float), np.asarray(mse, float), self.design, self.Gy)

    def sizes(self, x, mse):
        """[draws, layers] year-equal layer sizes; they add up to the year-equal weighted MSE."""
        return np.nanmean(self.per_year(x, mse), 1)


def correction_states(m, G, Cal, gamma, beta, kappa):
    """Field-mean residual x and case MSE of Base + gamma G + kappa Cal + beta S, from per-case sums (strength.MOMENTS);
    gamma, beta, kappa per case. Missing G or Cal count as 0."""
    m = np.asarray(m, float)
    n, sr, sS = m[:, 0], m[:, 1], m[:, 3]
    c = gamma * np.nan_to_num(G) + kappa * np.nan_to_num(Cal)
    srr, srS, sSS = m[:, 2], m[:, 4], m[:, 5]
    mse = (srr - 2 * beta * srS + beta ** 2 * sSS - 2 * c * (sr - beta * sS) + n * c ** 2) / n
    return sr / n - (c + beta * sS / n), mse


def subsets(components):
    """Every subset of the components, by size and then in the given order."""
    return [frozenset(s) for r in range(len(components) + 1) for s in itertools.combinations(components, r)]


def contributions(E, components):
    """Shapley contribution of each component to the reduction of every layer, in % of the Base total MSE.

    E: {subset: per-year layer values [draws, years, layers]} for every subset (the empty one is Base).
    Returns {component: [draws, layers]}.
    """
    n = len(components)
    T0 = np.nanmean(E[frozenset()].sum(2), 1)

    def red(a, b):
        return 100 * np.nanmean(a - b, 1) / T0[:, None]

    return {j: sum(factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n) * red(E[S], E[S | {j}])
                   for S in subsets(components) if j not in S) for j in components}


def correction_contributions(L, m, G, Cal, gamma, beta, kappa):
    """Shapley contributions of RPT (beta S), G (gamma G) and K (kappa Cal) for fixed per-case strengths.

    L: a Layers object of the cases; m, G, Cal as correction_states. Returns ({component: [draws, layers]}, per-year
    values of every subset)."""
    on = {'RPT': ('beta', beta), 'G': ('gamma', gamma), 'K': ('kappa', kappa)}
    E = {}
    for S in subsets(tuple(on)):
        k = {name: (v if c in S else 0 * np.asarray(v, float)) for c, (name, v) in on.items()}
        E[S] = L.per_year(*correction_states(m, G, Cal, k['gamma'], k['beta'], k['kappa']))
    return contributions(E, tuple(on)), E
