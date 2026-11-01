"""Case losses, severe harm and year-equal summaries with field-season bootstrap intervals.

Every summary weights the years equally and, within a year, each field-season once. Severe harm is a case MSE more than
10% above the Base. Intervals resample field-seasons with replacement within each year; all policies are evaluated on the
same draws through one count matrix.
"""
import numpy as np

SEVERE = 1.1
DRAWS = 2000


def scalar_rollup(year, weights, values):
    """Year-equal weighted mean of each column of values."""
    return np.mean([np.average(values[year == y], axis=0, weights=weights[year == y]) for y in np.unique(year)], 0)


def draw_matrix(fid_year, rng, draws=DRAWS):
    """Counts [1 + draws, field-seasons]: row 0 = every field-season once, rows 1.. = resamples within each year.

    fid_year: pandas Series year indexed by field-season id (sorted).
    """
    fids = fid_year.index.to_numpy()
    W = np.zeros((1 + draws, len(fids)), dtype=np.float32)
    W[0] = 1.
    for _, idx in fid_year.groupby(fid_year).groups.items():
        cols = np.searchsorted(fids, np.asarray(idx))
        pick = rng.integers(0, len(cols), size=(draws, len(cols)))
        flat = (np.arange(draws)[:, None] * len(cols) + pick).ravel()
        W[1:, cols] = np.bincount(flat, minlength=draws * len(cols)).reshape(draws, len(cols))
    return fids, W


def boot_rollup(W, col, year, weights, values):
    """Year-equal weighted means of every column for every draw: [draws, columns]. col: W column of each case."""
    out = np.zeros((len(W), values.shape[1]))
    count = np.zeros(len(W))
    for y in np.unique(year):
        r = year == y
        Wy = W[:, col[r]].astype(np.float64)
        den = Wy @ weights[r]
        good = den > 0
        out[good] += (Wy[good] @ (weights[r, None] * values[r])) / den[good, None]
        count += good
    return out / count[:, None]


def sign_p(v):
    """Two-sided bootstrap sign-tail p-value of a contrast with a one-count correction; v[0] is the point estimate, v[1:]
    the draws (the denominator counts all rows, as in the manuscript's analyses)."""
    v = np.asarray(v, float)
    return min(1., 2 * min((1 + np.sum(v[1:] <= 0)) / len(v), (1 + np.sum(v[1:] >= 0)) / len(v)))


def holm(p):
    """Holm-adjusted p-values of one comparison family."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    out = np.empty_like(adj)
    out[order] = adj
    return out


def summarize(losses, year, weights, col, W):
    """Index (100 x loss / Base) and severe-harm share (%) per policy column, point estimate and 95% interval.

    losses [cases, policies] with column 0 = Base.
    """
    v = np.concatenate([losses, losses > losses[:, [0]] * SEVERE + 1e-12], 1)
    stats = boot_rollup(W, col, year, weights, v)
    k = losses.shape[1]
    index, harm = 100 * stats[:, :k] / stats[:, [0]], 100 * stats[:, k:]
    ci = lambda x: np.percentile(x[1:], [2.5, 97.5], axis=0)  # noqa: E731
    return dict(index=index[0], index_ci=ci(index), harm=harm[0], harm_ci=ci(harm))
