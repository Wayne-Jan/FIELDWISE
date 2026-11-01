"""Residual pattern transfer (RPT): within-field corrections borrowed from reference fields.

For a query case q with reference cases d and weights a_qd (retrieval.py), every service pixel p of q receives
    h(p) = sum_d a_qd sum_j softmax_j(tau (A e_p) . (A e_j)) (r_j - rbar_d),
where j runs over the labelled pixels of d that have an embedding, e are field-centred pixel embeddings and r the Base
residuals. S = h minus its mean over the query's service pixels; S15 is S averaged over the pixels within 15 m (self
included) and re-centred. The projection A (D x D) and temperature tau are learned per target year on the legal history:
minimise the case MSE of Base + S over earlier-year queries, starting from A0 = diag(1 / sd) and tau0 = 1 / sqrt(D);
epochs are chosen on the latest development year, then refitted on all development years. The development queries are
the legal cases of the target year (selection.py); sd uses the embeddings of earlier-year cases decided by the cutoff.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from .selection import cutoffs, legal_mask, one_crop

BATCH, QS, MS, LR, MAX_EPOCHS, PATIENCE = 8, 1024, 512, 1e-3, 30, 5
VAL_QS, VAL_MS = 2048, 1024


def neighbours(lon, lat, radius=15.0):
    """Pixel neighbour lists within radius metres (self included), as CSR (indptr, indices) local to the field.

    lon, lat: pixel-centre longitude and latitude in degrees (WGS 84), converted to local metres around the field centre.
    With projected coordinates, build the lists directly (for example cKDTree.query_ball_point on x, y in metres).
    """
    lat0 = np.deg2rad(np.mean(lat))
    xy = np.column_stack([(lon - lon.mean()) * 111_320.0 * np.cos(lat0), (lat - lat.mean()) * 110_540.0])
    lst = cKDTree(xy).query_ball_point(xy, r=radius + 0.5)
    lens = np.fromiter((len(x) for x in lst), np.int64, len(lst))
    return np.concatenate([[0], np.cumsum(lens)]), np.concatenate([np.sort(x) for x in lst]).astype(np.int64)


@dataclass
class Model:
    A: np.ndarray
    tau: float
    A0: np.ndarray


class RPT:
    """Pixel data of one bank on a torch device.

    bank: fieldwise.bank.Bank whose case table has year, field, decision_ms, label_ms and pix0 (first pixel id of the
    case's field, so that pixel - pix0 is the service-pixel index). emb: [rows, D] float16 pixel embeddings and
    emb_index: table (case_id, row_start, n_pixels) of the embedding cases, contiguous and sorted by row_start.
    references: retrieval.select_references() output. geometry: optional {case_id: (indptr, indices)} 15 m neighbours.
    """

    def __init__(self, bank, emb, emb_index, references, geometry=None, device='cpu', seed=0):
        dev = self.dev = torch.device(device)
        torch.manual_seed(seed)
        self.gen = torch.Generator(device=dev).manual_seed(seed)
        C = self.cases = bank.cases.reset_index(drop=True)
        one_crop(C)
        I = emb_index.set_index('case_id')
        assert (I.loc[C.case_id, 'n_pixels'].to_numpy() == C.n_service.to_numpy()).all(), 'embedding / Base pixels differ'
        C['emb0'] = I.loc[C.case_id, 'row_start'].to_numpy()
        E = torch.empty(emb.shape, dtype=torch.float16, device=dev)
        for a in range(0, len(emb), 1 << 23):
            E[a:a + (1 << 23)] = torch.from_numpy(np.ascontiguousarray(emb[a:a + (1 << 23)])).to(dev)
        fin = torch.isfinite(E).all(1)
        E[~fin] = 0
        D, n_all = E.shape[1], len(E)
        seg = torch.zeros(n_all, dtype=torch.long, device=dev)
        starts = torch.tensor(emb_index.row_start.to_numpy(), device=dev)
        seg[starts[1:]] = 1
        seg = torch.cumsum(seg, 0)
        assert (np.diff(emb_index.row_start.to_numpy()) == emb_index.n_pixels.to_numpy()[:-1]).all()
        n_fin = torch.zeros(len(emb_index), device=dev).index_add_(0, seg, fin.float())
        means = torch.zeros(len(emb_index), D, device=dev).index_add_(0, seg, E.float()) / n_fin.clamp_min(1)[:, None]
        for a in range(0, n_all, 1 << 22):
            sl = slice(a, a + (1 << 22))
            E[sl] = torch.where(fin[sl, None], E[sl].float() - means[seg[sl]], torch.zeros(1, device=dev)).half()
        self.emb_pos = {c: k for k, c in enumerate(emb_index.case_id)}
        self.ss = torch.zeros(len(emb_index), D, device=dev).index_add_(0, seg, E.float() ** 2)
        self.n_fin, self.E, self.D = n_fin, E, D

        row = bank.row.astype(np.int64)
        local = bank.pixel - C.pix0.to_numpy()[row]
        assert (local >= 0).all() and (local < C.n_service.to_numpy()[row]).all()
        self.row = row
        self.emb_row = torch.from_numpy(C.emb0.to_numpy()[row] + local).to(dev)
        self.gt = torch.from_numpy(bank.gt_valid).to(dev)
        r = torch.from_numpy(bank.target.astype(np.float32) - bank.pred.astype(np.float32)).to(dev)
        self.r = torch.where(self.gt, r, torch.zeros_like(r))
        self.bounds = np.concatenate([[0], np.cumsum(np.bincount(row, minlength=len(C)))])
        assert (np.diff(self.bounds) == C.n_service.to_numpy()).all()
        self.rbar = torch.tensor(C.rbar.fillna(0).to_numpy(), dtype=torch.float32, device=dev)
        self.pfin = fin[self.emb_row]
        self.case_of = torch.from_numpy(row).to(dev)
        self.D_off, self.D_idx = self._csr(self.gt & self.pfin)        # reference side: labelled pixels with an embedding
        self.Q_off, self.Q_idx = self._csr(self.pfin)                  # query side: service pixels with an embedding

        self.groups = {q: (g.donor_row.to_numpy(), g.weight.to_numpy()) for q, g in references.groupby('query_row', sort=True)}
        self.queries = np.array(sorted(self.groups))
        kmax = max(len(v[0]) for v in self.groups.values())
        DON = np.full((len(self.queries), kmax), -1, np.int64)
        AW = np.zeros((len(self.queries), kmax), np.float32)
        for i, q in enumerate(self.queries):
            d, w = self.groups[q]
            DON[i, :len(d)], AW[i, :len(d)] = d, w
        self.DON, self.AW = torch.from_numpy(DON).to(dev), torch.from_numpy(AW).to(dev)
        self.n_gt = np.bincount(row[bank.gt_valid], minlength=len(C))
        key = C.year.astype(str) + '_' + C.field.astype(str)
        self.wfs = (1.0 / key.map(key.value_counts())).to_numpy()
        self.geometry = geometry

    def _csr(self, mask):
        idx = torch.nonzero(mask).squeeze(1)
        cnt = torch.bincount(self.case_of[idx], minlength=len(self.cases))
        return torch.cat([torch.zeros(1, dtype=torch.long, device=self.dev), torch.cumsum(cnt, 0)]), idx

    # ------------------------------------------------------------------ initial projection
    def initial(self, T, cutoff=None):
        """A0 = diag(1 / sd) of the embeddings of the earlier-year cases decided by the cutoff of T; None without history."""
        C = self.cases
        cut = cutoffs(C, cutoff)[T]
        hist = [self.emb_pos[c] for c in C.case_id[(C.year < T) & (C.decision_ms <= cut)]]
        if not hist:
            return None
        var = self.ss[hist].sum(0) / float(self.n_fin[hist].sum())
        return torch.diag((1.0 / var.clamp_min(1e-6)).sqrt())

    # ------------------------------------------------------------------ training
    def _sample(self, off, idx, rows, m):
        rows_c = rows.clamp_min(0)
        lo, n = off[rows_c], off[rows_c + 1] - off[rows_c]
        u = torch.rand((*rows.shape, m), device=self.dev, generator=self.gen)
        pick = lo[..., None] + torch.minimum((u * n[..., None]).long(), (n - 1).clamp_min(0)[..., None])
        ok = (rows[..., None] >= 0) & (n[..., None] > 0)
        return torch.where(ok, idx[pick.clamp_max(len(idx) - 1)], torch.full_like(pick, -1))

    def _batch_loss(self, qi, A, logt, qs, ms, dtype=torch.float32):
        """Sampled case MSE of Base + S (beta 1) for query positions qi: per-query losses and validity."""
        dev, E, er = self.dev, self.E, self.emb_row
        qp = self._sample(self.Q_off, self.Q_idx, torch.as_tensor(self.queries[qi], device=dev), qs)
        drows = self.DON[qi]
        dp = self._sample(self.D_off, self.D_idx, drows, ms)
        qv, dv = qp >= 0, dp >= 0
        eq = E[er[qp.clamp_min(0)]].to(dtype) @ A.T
        ed = E[er[dp.clamp_min(0)]].to(dtype) @ A.T
        s = torch.einsum('bqc,bkmc->bqkm', eq, ed) * logt.exp()
        p = torch.softmax(s.masked_fill(~dv[:, None], float('-inf')), -1).nan_to_num(0.0)
        rd = torch.where(dv, self.r[dp.clamp_min(0)] - self.rbar[drows.clamp_min(0)][..., None], torch.zeros(1, device=dev))
        h = torch.einsum('bqkm,bkm,bk->bq', p, rd.to(p.dtype), self.AW[qi].to(p.dtype))
        hm = (h * qv).sum(1, keepdim=True) / qv.sum(1, keepdim=True).clamp_min(1)
        S = torch.where(qv, h - hm, torch.zeros(1, device=dev))
        g = qv & self.gt[qp.clamp_min(0)]
        loss = (((self.r[qp.clamp_min(0)] - S) ** 2) * g).sum(1) / g.sum(1).clamp_min(1)
        return loss, g.any(1)

    @torch.no_grad()
    def _eval_sampled(self, val_q, wv, A, logt):
        """Validation loss with the same draws every epoch."""
        state = self.gen.get_state()
        self.gen.manual_seed(12345)
        tot, wsum = 0.0, 0.0
        for b in range(0, len(val_q), BATCH):
            loss, ok = self._batch_loss(val_q[b:b + BATCH], A, logt, VAL_QS, VAL_MS)
            ww = torch.as_tensor(wv[b:b + BATCH], dtype=torch.float32, device=self.dev) * ok
            tot += float((ww * loss).sum())
            wsum += float(ww.sum())
        self.gen.set_state(state)
        return tot / max(wsum, 1e-12)

    def _epochs(self, train_q, wq, A0, n_epochs, val_q=None, wv=None):
        """Adam from (A0, tau0); with validation: the parameters of the best epoch (epoch 0 = A0)."""
        dev = self.dev
        tau0 = 1.0 / np.sqrt(self.D)
        A = A0.clone().requires_grad_(True)
        logt = torch.tensor(float(np.log(tau0)), device=dev, requires_grad=True)
        opt = torch.optim.Adam([A, logt], lr=LR)
        tr_c, va_c, best, best_state, bad = [], [], np.inf, (A0.clone(), float(np.log(tau0))), 0
        w_t = torch.as_tensor(wq, dtype=torch.float32, device=dev)
        if val_q is not None:
            va_c.append(self._eval_sampled(val_q, wv, A.detach(), logt.detach()))
            best = va_c[0]
        for _ in range(n_epochs):
            perm = torch.randperm(len(train_q), device=dev, generator=self.gen).cpu().numpy()
            tot = 0.0
            for b in range(0, len(perm), BATCH):
                sel = perm[b:b + BATCH]
                loss, ok = self._batch_loss(train_q[sel], A, logt, QS, MS)
                ww = w_t[sel] * ok
                total = (ww * loss).sum() / ww.sum().clamp_min(1e-12)
                opt.zero_grad(set_to_none=True)
                total.backward()
                opt.step()
                tot += total.detach().item() * len(sel)
            tr_c.append(tot / len(perm))
            if val_q is not None:
                va_c.append(self._eval_sampled(val_q, wv, A.detach(), logt.detach()))
                if va_c[-1] < best - 1e-9:
                    best, best_state, bad = va_c[-1], (A.detach().clone(), float(logt)), 0
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        break
        if val_q is not None:
            return best_state[0], torch.tensor(best_state[1], device=dev), tr_c, va_c
        return A.detach(), logt.detach(), tr_c, va_c

    def _year_weights(self, qs):
        w = self.wfs[self.queries[qs]].astype(np.float64)
        y = self.cases.year.to_numpy()[self.queries[qs]]
        for v in np.unique(y):
            w[y == v] /= w[y == v].sum() * len(np.unique(y))
        return w

    def fit(self, T, cutoff=None):
        """Learned (A, tau) for target year T; A0 when fewer than two development years have queries."""
        A0 = self.initial(T, cutoff)
        C = self.cases
        qyear = C.year.to_numpy()[self.queries]
        devq = np.flatnonzero(legal_mask(C, T, cutoff)[self.queries] & (self.n_gt[self.queries] > 0))
        years = sorted(set(qyear[devq]))
        log = dict(year=int(T), dev_years=[int(v) for v in years], dev_queries=int(len(devq)))
        A, tau = A0, 1.0 / np.sqrt(self.D)
        if len(years) >= 2:
            tr, va = devq[qyear[devq] != years[-1]], devq[qyear[devq] == years[-1]]
            _, _, tr_c, va_c = self._epochs(tr, self._year_weights(tr), A0, MAX_EPOCHS, va, self._year_weights(va))
            e_star = int(np.argmin(va_c))
            log.update(inner_train=tr_c, inner_val=va_c, epochs=e_star)
            if e_star > 0:
                A, logt, _, _ = self._epochs(devq, self._year_weights(devq), A0, e_star)
                tau = float(logt.exp())
        log['tau'] = float(tau)
        return Model(A.cpu().numpy(), float(tau), A0.cpu().numpy()), log

    # ------------------------------------------------------------------ inference
    def _smooth(self, q, S0, fq):
        indptr, indices = self.geometry[self.cases.case_id.iloc[q]]
        n = len(S0)
        own = torch.repeat_interleave(torch.arange(n, device=self.dev), torch.from_numpy(np.diff(indptr)).to(self.dev))
        src = torch.from_numpy(indices).to(self.dev)
        f = fq[src].float()
        num = torch.zeros(n, device=self.dev).index_add_(0, own, S0[src] * f)
        den = torch.zeros(n, device=self.dev).index_add_(0, own, f)
        Ss = torch.where(fq & (den > 0), num / den.clamp_min(1), torch.zeros(1, device=self.dev))
        return torch.where(fq, Ss - Ss[fq].mean(), torch.zeros(1, device=self.dev)) if bool(fq.any()) else Ss

    @torch.no_grad()
    def infer(self, q, model):
        """S and S15 of query case q over its service pixels (S15 None without geometry)."""
        dev, E, er = self.dev, self.E, self.emb_row
        A = torch.as_tensor(model.A, device=dev)
        a, b = self.bounds[q], self.bounds[q + 1]
        Eq = E[er[a:b]].float() @ A.T
        h = torch.zeros(b - a, device=dev)
        for di, wd in zip(*self.groups.get(q, ((), ()))):
            lo, hi = int(self.D_off[di]), int(self.D_off[di + 1])
            if hi == lo:
                continue
            pix = self.D_idx[lo:hi]
            Ed = E[er[pix]].float() @ A.T
            rd = self.r[pix] - self.rbar[di]
            for c0 in range(0, b - a, 8192):
                s0 = Eq[c0:c0 + 8192] @ Ed.T
                s = torch.exp((s0 - s0.max(1, keepdim=True).values) * model.tau)
                h[c0:c0 + 8192] += float(wd) * (s @ rd) / s.sum(1)
        fq = self.pfin[a:b]
        S = torch.where(fq, h - h[fq].mean(), torch.zeros(1, device=dev)) if bool(fq.any()) else torch.zeros_like(h)
        S15 = self._smooth(q, S, fq) if self.geometry is not None else None
        return S, S15

    def sums(self, q, S):
        """Per-case sums on labelled pixels: n, sr, srr, sS, srS, sSS."""
        a, b = self.bounds[q], self.bounds[q + 1]
        m = self.gt[a:b]
        r, Sg = self.r[a:b][m], S[m]
        return dict(n=int(m.sum()), sr=float(r.sum()), srr=float((r * r).sum()), sS=float(Sg.sum()), srS=float((r * Sg).sum()),
                    sSS=float((Sg * Sg).sum()))


def run_year(rpt, T, model=None, cutoff=None):
    """Fit (unless model is given) and infer every query case of year T; returns (model, log, case sums, pixel S15)."""
    log = {}
    if model is None:
        model, log = rpt.fit(T, cutoff)
    rows, pix = [], {}
    for q in rpt.queries[rpt.cases.year.to_numpy()[rpt.queries] == T]:
        S, S15 = rpt.infer(int(q), model)
        out = S15 if S15 is not None else S
        rows.append(dict(row=int(q), **rpt.sums(int(q), out)))
        pix[int(q)] = out.cpu().numpy()
    return model, log, pd.DataFrame(rows), pix
