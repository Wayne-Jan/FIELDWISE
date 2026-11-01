"""Reference-field retrieval from field-level summaries.

Candidates for a query case are bank cases of the same decision stage from earlier years. A candidate is eligible when its
decision time is not later than the query's and its labels are available by the query's decision time. Distance: per
feature group, z-score with the statistics of the candidates decided by the earliest decision among the queries of the
same stage and year (inputs available to every one of them), and take the root mean square difference over the columns
finite in both cases; the group distances are averaged. The k nearest eligible candidates are weighted by inverse
distance.
"""
import numpy as np
import pandas as pd

from .selection import one_crop


def prefix_groups(columns, prefixes):
    """{group: columns starting with any of the group's prefixes}."""
    groups = {g: [c for c in columns if any(c.startswith(p) for p in ps)] for g, ps in prefixes.items()}
    empty = [g for g, cols in groups.items() if not cols]
    if empty:
        raise ValueError(f'feature groups without columns: {empty}')
    return groups


def balanced_distance(query, bank, groups, pool=None):
    """[queries, candidates] group-balanced distance (NaN where a group has no shared finite column).

    pool: boolean mask of the candidates whose statistics standardise the features (default: all candidates).
    """
    if query.empty or bank.empty:
        return np.full((len(query), len(bank)), np.nan)
    sub = slice(None) if pool is None or np.all(pool) else np.asarray(pool, bool)
    dists = []
    for columns in groups.values():
        q_raw = query[columns].to_numpy(np.float64)
        b_raw = bank[columns].to_numpy(np.float64)
        observed = np.isfinite(b_raw[sub]).any(axis=0)
        q_raw, b_raw = q_raw[:, observed], b_raw[:, observed]
        if q_raw.shape[1] == 0:
            return np.full((len(query), len(bank)), np.nan)
        q_ok, b_ok = np.isfinite(q_raw), np.isfinite(b_raw)
        with np.errstate(all='ignore'):
            mean, scale = np.nanmean(b_raw[sub], axis=0), np.nanstd(b_raw[sub], axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
        qz = np.where(q_ok, (q_raw - mean) / scale, 0.0)
        bz = np.where(b_ok, (b_raw - mean) / scale, 0.0)
        qm, bm = q_ok.astype(np.float64), b_ok.astype(np.float64)
        shared = qm @ bm.T
        sq = np.maximum((qz * qz) @ bm.T + qm @ (bz * bz).T - 2.0 * qz @ bz.T, 0.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            d = np.sqrt(sq / shared)
        d[shared <= 0] = np.nan
        dists.append(d)
    stacked = np.stack(dists)
    valid = np.isfinite(stacked).all(axis=0)
    out = np.mean(np.where(np.isfinite(stacked), stacked, 0.0), axis=0)
    out[~valid] = np.nan
    return out


def top_k(distance, k):
    """Indices of the k smallest finite distances per row (ties by candidate order) and their validity."""
    sortable = np.where(np.isfinite(distance), distance, np.inf)
    keys = np.broadcast_to(np.arange(distance.shape[1]), sortable.shape)
    order = np.lexsort((keys, sortable), axis=1)[:, :min(k, distance.shape[1])]
    return order, np.isfinite(np.take_along_axis(sortable, order, axis=1))


def inverse_distance_weights(distance, valid):
    """Normalised inverse-distance weights; candidates at distance 0 share the weight equally; invalid slots get 0."""
    w = np.zeros(distance.shape)
    zero = valid & (distance == 0.0)
    has_zero = zero.any(axis=1)
    if has_zero.any():
        w[has_zero] = zero[has_zero] / zero[has_zero].sum(axis=1, keepdims=True)
    rest = ~has_zero
    if rest.any():
        pos = valid[rest] & (distance[rest] > 0.0)
        raw = np.zeros_like(distance[rest])
        np.divide(1.0, distance[rest], out=raw, where=pos)
        tot = raw.sum(axis=1, keepdims=True)
        np.divide(raw, tot, out=raw, where=tot > 0.0)
        w[rest] = raw
    return w


def select_references(cases, features, groups, k=50):
    """Reference cases and weights for every case of the bank.

    cases: bank order, columns year, stage, decision_ms, label_ms. features: one row per case (same order) with the group
    columns. Returns a table (query_row, donor_row, weight) with bank row indices.
    """
    one_crop(cases)
    feats = features.reset_index(drop=True)
    year, stage = cases.year.to_numpy(), cases.stage.to_numpy()
    dec, lab = cases.decision_ms.to_numpy(), cases.label_ms.to_numpy()
    parts = []
    for (s, T), g in cases.reset_index(drop=True).groupby(['stage', 'year']):
        qi = g.index.to_numpy()
        di = np.flatnonzero((stage == s) & (year < T))
        if len(di) == 0:
            continue
        d = balanced_distance(feats.iloc[qi], feats.iloc[di], groups, pool=dec[di] <= dec[qi].min())
        ok = ((year[di][None, :] < year[qi][:, None]) & (dec[di][None, :] <= dec[qi][:, None])
              & (lab[di][None, :] <= dec[qi][:, None]))
        d = np.where(ok, d, np.nan)
        order, valid = top_k(d, k)
        w = inverse_distance_weights(np.take_along_axis(d, order, 1), valid)
        qq, jj = np.nonzero(valid & (w > 0))
        parts.append(pd.DataFrame(dict(query_row=qi[qq], donor_row=di[order[qq, jj]], weight=w[qq, jj])))
    return pd.concat(parts, ignore_index=True)
