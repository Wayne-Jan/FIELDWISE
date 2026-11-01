"""Weighted ridge regression and standardisation used by the field-level models."""
import numpy as np


def fit_scaler(Xtr, wtr):
    """Weighted mean and standard deviation of the finite training values per column, and which columns have any."""
    ok = np.isfinite(Xtr)
    wv = wtr[:, None] * ok
    tot = wv.sum(0)
    has = tot > 0
    mu = np.where(has, np.nansum(wv * np.nan_to_num(Xtr), 0) / np.where(has, tot, 1), 0.)
    var = np.where(has, (wv * np.nan_to_num(Xtr - mu) ** 2).sum(0) / np.where(has, tot, 1), 1.)
    return mu, np.where(var > 0, np.sqrt(var), 1.), has


def apply_scaler(X, scaler):
    """Z-scores with a fitted scaler; missing values and columns without training values become 0."""
    mu, sd, has = scaler
    Z = np.nan_to_num((X - mu) / sd, nan=0.)
    Z[:, ~has] = 0.
    return Z


def standardize(Xtr, wtr, *others):
    """Z-score with weighted training-row statistics; missing values become 0 (the training mean).

    Columns without any finite training value are set to 0 everywhere. Returns the standardised training matrix followed by
    the other matrices, standardised with the same statistics.
    """
    scaler = fit_scaler(Xtr, wtr)
    return [apply_scaler(X, scaler) for X in (Xtr, *others)]


def ridge_path(X, y, w, Xq, lams):
    """Weighted centred ridge with a free intercept for every penalty, from one eigendecomposition.

    Returns predictions at Xq [rows, penalties] and coefficients [features, penalties].
    """
    sw = w / w.sum()
    xm, ym = sw @ X, sw @ y
    Xc = X - xm
    s, V = np.linalg.eigh(Xc.T @ (w[:, None] * Xc))
    beta = V @ ((V.T @ (Xc.T @ (w * (y - ym))))[:, None] / (s[:, None] + np.asarray(lams)[None, :]))
    return (Xq - xm) @ beta + ym, beta


def ridge_two_block(X, y, w, Xq, n_first, pairs, return_center=False):
    """Weighted centred ridge with separate penalties for the first n_first columns and the remaining columns.

    pairs: sequence of (lambda_first, lambda_second); lambda_second = inf drops the second block. Returns predictions
    [rows, pairs] and coefficients [features, pairs]; with return_center also the weighted means of X and y, so that a
    prediction is (x - x_mean) @ beta + y_mean.
    """
    sw = w / w.sum()
    xm, ym = sw @ X, sw @ y
    Xc = X - xm
    A, b = Xc.T @ (w[:, None] * Xc), Xc.T @ (w * (y - ym))
    p = X.shape[1]
    beta = np.zeros((len(pairs), p))
    fin = np.array([np.isfinite(l2) for _, l2 in pairs])
    if fin.any():
        P = np.array([np.r_[np.full(n_first, l1), np.full(p - n_first, l2)] for (l1, l2), f in zip(pairs, fin) if f])
        beta[fin] = np.linalg.solve(A[None] + P[:, :, None] * np.eye(p)[None],
                                    np.broadcast_to(b, (len(P), p))[..., None])[..., 0]
    if (~fin).any():
        l1 = np.array([l1 for (l1, _), f in zip(pairs, fin) if not f])
        A1, b1 = A[:n_first, :n_first], b[:n_first]
        beta[~fin, :n_first] = np.linalg.solve(A1[None] + l1[:, None, None] * np.eye(n_first)[None],
                                               np.broadcast_to(b1, (len(l1), n_first))[..., None])[..., 0]
    if return_center:
        return (Xq - xm) @ beta.T + ym, beta.T, xm, ym
    return (Xq - xm) @ beta.T + ym, beta.T
