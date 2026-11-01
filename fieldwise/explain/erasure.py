"""Concept erasure with LEACE (Belrose et al., 2023) and matched random controls (manuscript Sections 4.6.3 and 4.7.3,
Appendices C.3 and D.2).

LEACE. With weighted, centred features X and concept Z on the fitting rows, C = cov(X), S = cov(X, Z), whitening
W = C^(-1/2) and U an orthonormal basis of the column space of W S, the erased features are X - (X - mu) W U U' W^(-1). After
erasure the cross-covariance with Z vanishes on the fitting rows. Fitting weights: years equal, rows equal within a year.

Random controls have the rank of the named erasure and the same expected squared perturbation on the fitting rows
(tr(U' C U)): a Haar-random subspace in the whitened eigen-coordinates is mixed towards the extreme eigenvectors until the
perturbation matches.

Field-mean erasure. Maps are fitted on the legal history of each FM, crop, target year and decision time and act on the
z-scored FM means and window differences (the missing-window flags are kept). Frozen readouts keep their penalties and
coefficients and see erased query features; refitted readouts reselect the penalties by leave-one-year-out after erasing the
fitting and query features, each inner fold with its own map.
"""
from dataclasses import dataclass

import numpy as np

from ..readout import COMBOS, _scale, _scaler

SEEDS = tuple(range(41, 49))


def standardize(X, fit):
    return _scale(X, _scaler(X, fit))


def whiten(C):
    ev, V = np.linalg.eigh((C + C.T) / 2)
    keep = ev > max(ev[-1], 0) * 128 * np.finfo(float).eps
    ev, V = ev[keep], V[:, keep]
    return ev, V, (V / np.sqrt(ev)) @ V.T, (V * np.sqrt(ev)) @ V.T


def leace(C, Sxz):
    """Rank-k weighted LEACE: (ev, V, W, Wi, U) with U an orthonormal basis of col(W Sxz)."""
    ev, V, W, Wi = whiten(C)
    u, s, _ = np.linalg.svd(W @ Sxz, full_matrices=False)
    r = int((s > (s[0] if len(s) else 0) * 128 * np.finfo(float).eps).sum()) if len(s) and s[0] > 1e-12 else 0
    return ev, V, W, Wi, u[:, :r]


def matched_random(ev, V, r, target, seed):
    """Random rank-r subspace with tr(Z' diag(ev) Z) = target: Haar draw Z mixed towards the r extreme eigenvectors (top
    if the target is larger, bottom if smaller), Z(a) = orth(Z + a E), a by bisection."""
    m = len(ev)
    rng = np.random.default_rng(seed)
    Z = np.linalg.qr(rng.normal(size=(m, r)))[0]
    d = lambda Y: float(np.einsum('ij,i,ij->', Y, ev, Y))  # noqa: E731
    d0 = d(Z)
    if abs(d0 - target) <= 1e-12 * max(abs(target), 1e-30):
        return V @ Z
    order = np.argsort(-ev) if target > d0 else np.argsort(ev)
    E = np.eye(m)[:, order[:r]]
    f = lambda a: d(np.linalg.qr(Z + a * E)[0]) - target  # noqa: E731
    lo, hi = 0.0, 1.0
    while np.sign(f(hi)) == np.sign(f(lo)):
        hi *= 2
        assert hi < 1e12, 'cannot match the concept distortion'
    for _ in range(200):
        mid = (lo + hi) / 2
        if np.sign(f(mid)) == np.sign(f(lo)):
            lo = mid
        else:
            hi = mid
        if abs(f(mid)) <= 1e-10 * abs(target):
            break
    Zm = np.linalg.qr(Z + mid * E)[0]
    assert abs(d(Zm) - target) <= 1e-7 * abs(target), (d(Zm), target)
    return V @ Zm


def ridge_r2(Mx, Sxz, Szz):
    """In-sample R^2 of every concept column from a ridge (0.01 on z-scored features) on centred moments."""
    sd = np.sqrt(np.maximum(np.diag(Mx), 1e-12))
    Cs = Mx / sd[:, None] / sd[None, :]
    B = np.linalg.solve(Cs + 0.01 * np.eye(len(sd)), Sxz / sd[:, None])
    fitted = np.einsum('pk,pq,qk->k', B, Cs, B)
    cross = np.einsum('pk,pk->k', B, Sxz / sd[:, None])
    var = np.diag(Szz)
    resid = var - 2 * cross + fitted
    return np.where(var > 1e-12, 1 - resid / np.where(var > 1e-12, var, 1), np.nan)


def year_weights(year):
    """Weights of the fitting rows: years equal, rows equal within a year."""
    years, cnt = np.unique(year, return_counts=True)
    return np.array([1 / len(years) / cnt[np.searchsorted(years, v)] for v in year])


@dataclass
class Eraser:
    mu: np.ndarray
    R: np.ndarray
    L: np.ndarray
    rank: int
    target: float

    def change(self, x):
        return (x - self.mu) @ self.R @ self.L

    def __call__(self, x):
        return x - self.change(x)


def fit_erasers(x, Z, fit, year, seeds=SEEDS):
    """Named eraser and matched random erasers of features x [cases, p] for concept Z [cases, q] (z-scored on the fitting
    rows, missing values at the training mean), fitted on rows fit. Returns ({name: Eraser}, moments (C, S, Szz))."""
    z = standardize(Z, fit)
    w = year_weights(year[fit])
    mu, mz = w @ x[fit], w @ z[fit]
    xc, zc = x[fit] - mu, z[fit] - mz
    C, S, Szz = xc.T @ (w[:, None] * xc), xc.T @ (w[:, None] * zc), zc.T @ (w[:, None] * zc)
    ev, V, W, Wi, U = leace(C, S)
    r = U.shape[1]
    target = float(np.trace(U.T @ C @ U))
    bases = [('named', U)] + [(f'random{s}', matched_random(ev, V, r, target, s) if r else U) for s in seeds]
    out = {}
    for name, u in bases:
        L = u.T @ Wi
        e = Eraser(mu, W @ u, L, r, target)
        assert abs(float(np.trace(L @ L.T)) - target) <= 1e-7 * max(target, 1e-30)
        out[name] = e
    return out, (C, S, Szz)


def spectral(F, fit, pred):
    """Factors of the FM block for the exact two-block ridge path (eigenvectors of the Gram or dual Gram matrix)."""
    f, fq = F[fit], F[pred]
    dual = len(fit) < F.shape[1]
    G = f @ f.T if dual else f.T @ f
    ev, V = np.linalg.eigh((G + G.T) / 2)
    ev = np.maximum(ev, 0.)
    return ev, (V if dual else f @ V), (fq @ f.T @ V if dual else fq @ V), dual


def ridge_path(B, y, fit, pred, factors, combos=COMBOS):
    """Two-block ridge predictions [combinations, pred rows] (shared block B with penalty combos[:, 0], FM block with
    combos[:, 1]); the FM block enters through its spectral factors, leaving only small solves for the shared block."""
    ev, T, Q, dual = factors
    b, bq = B[fit], B[pred]
    ym = y[fit].mean()
    yc = y[fit] - ym
    tb, ty = T.T @ b, T.T @ yc
    inv = 1 / (ev[None, :] + combos[:, 1, None])
    h = inv * ev if dual else inv
    A = (b.T @ b)[None] - np.einsum('ki,ij,il->kjl', h, tb, tb) + combos[:, 0, None, None] * np.eye(b.shape[1])[None]
    rhs = (b.T @ yc)[None] - np.einsum('ki,ij,i->kj', h, tb, ty)
    beta = np.linalg.solve(A, rhs[..., None])[..., 0]
    base = ym + np.einsum('qi,ki,i->kq', Q, inv, ty)
    adj = bq[None] - np.einsum('qi,ki,ij->kqj', Q, inv, tb)
    return base + np.einsum('kqj,kj->kq', adj, beta)


def readout_erasure(XB, XF, Y, Z, year, train, query, continuous, seeds=SEEDS):
    """Original, frozen and refitted readouts of the query rows after named and random erasure, for one FM.

    XB, Y: {label: shared block, residual target} (several Bases share the erasure maps); XF: FM block; Z: concept;
    continuous: FM columns erased (the others are missing flags). Returns (predictions {(label, mode, arm): rhat}, penalties
    {(label, arm): index into COMBOS}, outer erasers).
    """
    folds = [(str(v), train[year[train] != v], train[year[train] == v]) for v in np.unique(year[train])] + [('outer', train, query)]
    scores, outer, frozen, coef, pick = {}, {}, {}, {}, {}
    erasers_outer = None
    for fold, fit, va in folds:
        F = standardize(XF, fit)
        Bs = {lab: standardize(XB[lab], fit) for lab in XB}
        fac = spectral(F, fit, va)
        for lab in XB:
            pred = ridge_path(Bs[lab], Y[lab], fit, va, fac)
            if fold != 'outer':
                scores.setdefault((lab, 'original'), []).append(np.mean((pred - Y[lab][va]) ** 2, 1))
            else:
                k = int(np.argmin(np.mean(scores[lab, 'original'], 0)))
                pick[lab, 'original'], outer[lab, 'original'] = k, pred[k]
                z = np.column_stack([Bs[lab], F])
                zf = z[fit]
                ym = Y[lab][fit].mean()
                pen = np.r_[np.full(Bs[lab].shape[1], COMBOS[k, 0]), np.full(F.shape[1], COMBOS[k, 1])]
                coef[lab] = (np.linalg.solve(zf.T @ zf + np.diag(pen), zf.T @ (Y[lab][fit] - ym)), ym)
        erasers, _ = fit_erasers(F[:, continuous], Z, fit, year, seeds)
        if fold == 'outer':
            erasers_outer = erasers
        for arm, er in erasers.items():
            Fe = F.copy()
            Fe[:, continuous] = er(F[:, continuous])
            factors = spectral(standardize(Fe, fit), fit, va)
            for lab in XB:
                pred = ridge_path(Bs[lab], Y[lab], fit, va, factors)
                if fold != 'outer':
                    scores.setdefault((lab, arm), []).append(np.mean((pred - Y[lab][va]) ** 2, 1))
                else:
                    k = int(np.argmin(np.mean(scores[lab, arm], 0)))
                    pick[lab, arm], outer[lab, arm] = k, pred[k]
                    co, ym = coef[lab]
                    frozen[lab, arm] = ym + np.column_stack([Bs[lab][query], Fe[query]]) @ co
    preds = {(lab, 'refit', arm): v for (lab, arm), v in outer.items()}
    preds.update({(lab, 'frozen', arm): v for (lab, arm), v in frozen.items()})
    return preds, pick, erasers_outer
