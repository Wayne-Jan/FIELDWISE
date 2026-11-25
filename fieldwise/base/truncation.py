"""Random-cutoff truncation for truncated-season Base training (manuscript Section 4.1, Fig. 3A).

At the start of each epoch, every training field-season draws one cutoff day W uniformly over the integers in [W_lo, W_hi],
where W_lo is its earliest in-season window end (tau1 when present) and W_hi is its harvest window end. All pixels of that
field-season are truncated at W for that epoch (Sentinel-2 steps and daily weather with day < W). The returned view shares
all underlying tensors and only replaces per-sample lengths (s_len) and per-case weather dates (wd), so both Transformer
and 3D-LSTM batch functions work unchanged.
"""
import numpy as np
import torch

SALT = 20261111


class TruncView:
    """Zero-copy view of PixelData with per-sample s_len and per-case wd replaced."""

    def __init__(self, p, s_len, wd):
        self._p, self.s_len, self.wd = p, s_len, wd

    def __getattr__(self, name):
        return getattr(self._p, name)


class CaseClock:
    """Cutoff bounds [lo, hi] (UTC day numbers) of training field-seasons and per-epoch truncated views."""

    def __init__(self, p, fit_idx, lo_days, hi_days, date_pad=-1000):
        """
        p: PixelData of training (whole-season) cases with absolute UTC day numbers in p.day and p.wd.
        fit_idx: 1D tensor of sample indices in p belonging to the training split.
        lo_days, hi_days: arrays indexed by case position in p (len == len(p.case_ids)), inclusive UTC day bounds.
        """
        dev = p.s_len.device
        cases = torch.unique(p.s_case[fit_idx]).cpu().numpy()
        lo = np.asarray(lo_days, np.int64)[cases]
        hi = np.asarray(hi_days, np.int64)[cases]
        assert (lo <= hi).all(), 'truncation lower bound exceeds upper bound'
        self.lo, self.hi = lo, hi
        self.p, self.fit_idx, self.date_pad = p, fit_idx, date_pad
        self.cases = torch.as_tensor(cases, dtype=torch.long, device=dev)

        first = torch.full((len(p.case_ids),), -1, dtype=torch.long, device=dev)
        first.scatter_reduce_(0, p.s_case[fit_idx], fit_idx, reduce='amin', include_self=False)
        f = first[self.cases]
        L = p.s_len[f]
        Lmax = int(L.max().item()) if len(L) else 0
        ar = torch.arange(Lmax, device=dev)
        g = (p.s_start[f][:, None] + ar[None, :]).clamp(max=max(len(p.day) - 1, 0))
        if Lmax > 0 and len(p.day) > 0:
            days = torch.where(
                ar[None, :] < L[:, None],
                p.day[g].to(torch.float64),
                torch.full((len(f), Lmax), float('inf'), device=dev, dtype=torch.float64),
            )
            assert bool((days[:, 1:] >= days[:, :-1]).all()), 'Sentinel-2 step days must be non-decreasing'
        else:
            days = torch.empty((len(f), 0), device=dev, dtype=torch.float64)
        self.days, self.L = days, L

        pos = torch.full((len(p.case_ids),), -1, dtype=torch.long, device=dev)
        pos[self.cases] = torch.arange(len(cases), device=dev)
        self.fit_pos = pos[p.s_case[fit_idx]]

    def cutoffs(self, seed, epoch):
        """Draw one integer UTC cutoff day per training field-season in [lo, hi]."""
        rng = np.random.default_rng([int(seed), SALT, int(epoch)])
        return rng.integers(self.lo, self.hi + 1)

    def view(self, seed, epoch, W=None):
        """Return (TruncView, summary dict) for (seed, epoch); optional W overrides the random draw."""
        p = self.p
        dev = p.s_len.device
        W = self.cutoffs(seed, epoch) if W is None else np.asarray(W, np.int64)
        assert ((W >= self.lo) & (W <= self.hi)).all(), 'cutoff day outside [lo, hi]'
        wt = torch.as_tensor(W, dtype=torch.float64, device=dev)
        if self.days.shape[1] > 0:
            Lc = torch.searchsorted(self.days, wt[:, None], right=False)[:, 0]
            Lc = torch.minimum(Lc, self.L)
        else:
            Lc = torch.zeros_like(self.L)
        s_len = p.s_len.clone()
        s_len[self.fit_idx] = Lc[self.fit_pos]
        wd = p.wd.clone()
        rows = wd[self.cases]
        cut = (rows != self.date_pad) & (rows >= wt.to(rows.dtype)[:, None])
        wd[self.cases] = torch.where(cut, torch.full_like(rows, self.date_pad), rows)
        info = dict(
            cases=int(len(W)),
            span_days_mean=float((self.hi - self.lo).mean()) if len(W) else 0.0,
            step_share_mean=float((Lc.double() / self.L.clamp(min=1).double()).mean()) if len(W) else 0.0,
        )
        return TruncView(p, s_len, wd), info
