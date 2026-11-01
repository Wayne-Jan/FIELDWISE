"""Residual bank: out-of-sample Base predictions and labels of every case, pixel by pixel.

A bank holds a case table (one row per case, in pixel order) and pixel arrays: row (case index), pixel (service-pixel id),
pred (Base prediction), target (label) and gt_valid (labelled pixel). The residual of a pixel is target - pred; rbar is the
mean residual over the labelled pixels of a case.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PIXEL_KEYS = ('row', 'pixel', 'pred', 'target', 'gt_valid')


@dataclass
class Bank:
    cases: pd.DataFrame
    row: np.ndarray
    pixel: np.ndarray
    pred: np.ndarray
    target: np.ndarray
    gt_valid: np.ndarray

    @classmethod
    def load(cls, folder, name):
        folder = Path(folder)
        z = np.load(folder / f'{name}_pixels.npz')
        return cls(pd.read_parquet(folder / f'{name}_cases.parquet'), *(z[k] for k in PIXEL_KEYS))

    def save(self, folder, name):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.cases.to_parquet(folder / f'{name}_cases.parquet', index=False)
        np.savez(folder / f'{name}_pixels.npz', **{k: getattr(self, k) for k in PIXEL_KEYS})

    @property
    def bounds(self):
        """Pixel offsets of every case: case i owns pixels bounds[i]:bounds[i + 1]."""
        return np.concatenate([[0], np.cumsum(np.bincount(self.row.astype(np.int64), minlength=len(self.cases)))])

    def moments(self):
        """Per-case sums on labelled pixels and Base statistics on all service pixels (float64).

        Columns: n (labelled pixels), sy, sp, syy, syp, spp (means over labelled pixels), sr, srr (sums of the residual and
        its square), rbar, mse, n_service, p_mean, p_sd, log_n.
        """
        row, gt = self.row.astype(np.int64), self.gt_valid
        y, p = self.target.astype(np.float64), self.pred.astype(np.float64)
        nc = len(self.cases)

        def total(v=None, m=None):
            r = row if m is None else row[m]
            vv = None if v is None else (v if m is None else v[m])
            return np.bincount(r, weights=vv, minlength=nc)

        n = total(None, gt)
        M = pd.DataFrame(index=self.cases.index)
        for k, v in dict(sy=y, sp=p, syy=y * y, syp=y * p, spp=p * p).items():
            M[k] = total(v, gt) / np.maximum(n, 1)
        r = y - p
        M['n'] = n
        M['sr'] = total(r, gt)
        M['srr'] = total(r * r, gt)
        M['rbar'] = M.sy - M.sp
        M['mse'] = M.syy - 2 * M.syp + M.spp
        ns = total()
        M['n_service'] = ns
        M['p_mean'] = total(p) / ns
        M['p_sd'] = np.sqrt(np.maximum(total(p * p) / ns - M.p_mean ** 2, 0))
        M['log_n'] = np.log(ns)
        return M


def case_table(case_id, row, pred, target, gt_valid):
    """Case statistics of a bank in the order of case_id (row indexes case_id)."""
    row = row.astype(np.int64)
    nb = len(case_id)
    pred, tgt, gt = pred.astype(np.float64), target.astype(np.float64), gt_valid & np.isfinite(target)
    n_s = np.bincount(row, minlength=nb)
    n_g = np.bincount(row[gt], minlength=nb)
    C = pd.DataFrame(dict(case_id=case_id, n_service=n_s, n_gt=n_g))
    C['gt_mean'] = np.bincount(row[gt], weights=tgt[gt], minlength=nb) / np.maximum(n_g, 1)
    C['pred_mean_gt'] = np.bincount(row[gt], weights=pred[gt], minlength=nb) / np.maximum(n_g, 1)
    C['rbar'] = C.gt_mean - C.pred_mean_gt
    C['pred_mean_service'] = np.bincount(row, weights=pred, minlength=nb) / np.maximum(n_s, 1)
    C.loc[n_g == 0, ['gt_mean', 'pred_mean_gt', 'rbar']] = np.nan
    return C


def ensemble(banks):
    """Bank of the mean prediction of several Base runs over the same cases and pixels (e.g. five seeds)."""
    b0 = banks[0]
    for b in banks[1:]:
        assert np.array_equal(b.cases.case_id.to_numpy(), b0.cases.case_id.to_numpy()), 'case order differs'
        for k in ('row', 'pixel', 'gt_valid'):
            assert np.array_equal(getattr(b, k), getattr(b0, k)), f'{k} differs'
        assert np.array_equal(b.target[b0.gt_valid], b0.target[b0.gt_valid]), 'labels differ'
    pred = np.mean([b.pred.astype(np.float64) for b in banks], axis=0)
    stats = case_table(b0.cases.case_id.to_numpy(), b0.row, pred, b0.target, b0.gt_valid)
    keep = [c for c in b0.cases.columns if c not in stats.columns]
    cases = pd.concat([b0.cases[keep].reset_index(drop=True), stats.drop(columns='case_id')], axis=1)
    cases.insert(0, 'case_id', stats.case_id)
    return Bank(cases, b0.row, b0.pixel, pred.astype(np.float32), b0.target, b0.gt_valid)
