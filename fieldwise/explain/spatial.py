"""Spatial assessment of within-field predictions (manuscript Section 4.5.3, Appendix C.1).

Edge contrast. Distances run from pixel centres to the boundary of the field's service-pixel footprint. A case's contrast
is the mean over valid pixels 10 <= d < 20 m minus the mean over d >= 40 m, for observed yield or a prediction; its
contrast error is |contrast(prediction) - contrast(observed)|. Cases need valid pixels in both zones.

Within-field response. Per case, the slope of the prediction on observed yield, Cov(y, prediction) / Var(y), over valid
pixels; cases without observed variance are left out.

Intervals resample whole clusters (fields or coordinate-matched land groups) with all their seasons and decision times,
and recompute the evaluation weights in every draw (cluster_weights).
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EDGE = (10.0, 20.0)
INTERIOR = 40.0
_SIDES = ((-1, 0, (-.5, -.5), (-.5, .5)), (1, 0, (.5, -.5), (.5, .5)), (0, -1, (-.5, -.5), (.5, -.5)), (0, 1, (-.5, .5), (.5, .5)))


def boundary_distance(row, col, pixel_size=10.0):
    """Exact distance (m) from every pixel centre to the boundary of the union of the pixels' grid cells.

    row, col: grid indices of all service pixels of one field (north-up square grid of pixel_size metres).
    """
    r = np.asarray(row) - np.min(row) + 1
    c = np.asarray(col) - np.min(col) + 1
    mask = np.zeros((r.max() + 2, c.max() + 2), bool)
    mask[r, c] = True
    seg = []
    for dr, dc, a, b in _SIDES:                                                 # cell sides without a neighbouring cell
        ok = ~mask[r + dr, c + dc]
        v = np.column_stack([r[ok], c[ok]]).astype(float)
        seg.append(np.stack([v + a, v + b], 1))
    seg = np.concatenate(seg)
    p = np.column_stack([r, c]).astype(float)
    tree = cKDTree(seg.mean(1))
    d0 = tree.query(p)[0]
    # the nearest side has its midpoint within (its distance + 1/2) <= (nearest midpoint distance + 1/2)
    cand = tree.query_ball_point(p, d0 + 0.5 + 1e-9)
    lens = np.fromiter((len(x) for x in cand), np.int64, len(cand))
    idx = np.concatenate(cand).astype(np.int64)
    own = np.repeat(np.arange(len(p)), lens)
    a = seg[idx, 0]
    v = seg[idx, 1] - a
    t = np.clip(((p[own] - a) * v).sum(1) / (v * v).sum(1), 0, 1)
    d = np.linalg.norm(p[own] - a - t[:, None] * v, axis=1)
    return np.minimum.reduceat(d, np.concatenate([[0], np.cumsum(lens)[:-1]])) * pixel_size


def _case_mean(z, keep, case, n):
    num = np.bincount(case[keep], weights=z[keep], minlength=n)
    cnt = np.bincount(case[keep], minlength=n)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(cnt > 0, num / cnt, np.nan), cnt


def edge_contrast(z, dist, valid, case, n_cases):
    """[cases] edge-band minus interior mean of z over valid pixels (NaN without pixels in both zones).

    z, dist, valid, case: per pixel (case = case index 0 .. n_cases - 1)."""
    z, dist = np.asarray(z, float), np.asarray(dist, float)
    edge, ne = _case_mean(z, valid & (dist >= EDGE[0]) & (dist < EDGE[1]), case, n_cases)
    inner, ni = _case_mean(z, valid & (dist >= INTERIOR), case, n_cases)
    return np.where((ne > 0) & (ni > 0), edge - inner, np.nan)


def response_slope(y, pred, valid, case, n_cases):
    """[cases] Cov(y, pred) / Var(y) over valid pixels (population moments; NaN when Var(y) = 0)."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ym, n = _case_mean(y, valid, case, n_cases)
    pm, _ = _case_mean(pred, valid, case, n_cases)
    dy, dp = y - ym[case], pred - pm[case]
    cov, _ = _case_mean(dy * dp, valid, case, n_cases)
    var, _ = _case_mean(dy * dy, valid, case, n_cases)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(var > 0, cov / var, np.nan)


def cluster_weights(cluster, year, fid, rng, draws=2000):
    """[1 + draws, cases] weights. Row 0: evaluation weights (years equal; field-seasons equal within a year; a field-season's
    weight split over its cases). Rows 1..: clusters drawn with replacement (multinomial over the clusters in order of first
    appearance), the same split, renormalised within each year."""
    codes, u = pd.factorize(pd.Series(cluster))
    z = np.vstack([np.ones(len(u)), rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=draws)]).astype(float)
    V = z[:, codes] / pd.Series(fid).groupby(np.asarray(fid)).transform('size').to_numpy()[None, :]
    year = np.asarray(year)
    ny = len(pd.unique(year))
    for y in pd.unique(year):
        ix = year == y
        den = V[:, ix].sum(1)
        V[:, ix] = np.divide(V[:, ix], den[:, None], out=np.zeros_like(V[:, ix]), where=den[:, None] > 0) / ny
    return V


def weighted(V, x):
    """[draws] weighted means of x over its finite entries."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    return (V[:, ok] @ x[ok]) / V[:, ok].sum(1)
