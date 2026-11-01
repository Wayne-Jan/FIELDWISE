"""Leakage tests of the fitting chain of fieldwise.tutorial.run (residual bank -> references -> RPT -> G, Cal, readouts,
strengths, alpha) on synthetic data: the fits, choices, references and predictions serving the target year do not change
when labels it may not use change, namely its own labels and earlier labels that arrive after its decisions."""
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from fieldwise import bank as fb, calibration, decision, environment, readout, retrieval, rpt, selection, strength

DAY = 86_400_000
STAGES = ('tau1', 'tau2')
TARGET = 2022
FAST_RPT = dict(QS=64, MS=32, VAL_QS=64, VAL_MS=32, MAX_EPOCHS=6)        # small RPT samples keep the test quick


def study(rng, years=(2018, 2019, 2020, 2021, 2022), fields=6, side=(4, 6), dim=6):
    """Synthetic field-seasons with two decision stages; both cases of a field-season serve its pixels."""
    npx = side[0] * side[1]
    rows = []
    for y in years:
        for f in range(fields):
            fs = len(rows) // len(STAGES)
            for k, s in enumerate(STAGES):
                dec = (y - 1970) * 365 * DAY + k * 30 * DAY
                rows.append(dict(case_id=f'{f}:{y}:{s}', year=y, field=f, stage=s, decision_ms=dec, label_ms=dec + 200 * DAY,
                                 fs=fs, pix0=fs * npx, n_service=npx))
    C = pd.DataFrame(rows)
    late = ((C.year == TARGET - 1) & (C.field == 0)).to_numpy()            # labels after every decision of the target year
    C.loc[late, 'label_ms'] = C[C.year == TARGET].decision_ms.max() + DAY
    n_fs = int(C.fs.nunique())
    E_fs = rng.normal(size=(n_fs, npx, dim))
    emb = (E_fs[C.fs.to_numpy()] + 0.1 * rng.normal(size=(len(C), npx, dim))).reshape(-1, dim).astype(np.float16)
    target = (8 + rng.normal(0, 1, n_fs)[:, None] + 0.6 * E_fs[..., 0] + rng.normal(0, .3, (n_fs, npx))).ravel()
    rr, cc = np.divmod(np.arange(npx), side[1])
    nb = rpt.neighbours(120.2 + cc * 1e-4, 23.0 - rr * 9e-5)
    feats = pd.DataFrame(rng.normal(size=(len(C), 3)), columns=['q_a__x', 'q_a__y', 'q_b__z'])
    fs_year = C.drop_duplicates('fs').year.to_numpy()
    daily = rng.normal(size=(n_fs, 300, 2))
    return dict(cases=C, late=late, cutoff=selection.earliest_decision(C).to_dict(),
                row=np.repeat(np.arange(len(C)), npx), pix=(C.pix0.to_numpy()[:, None] + np.arange(npx)).ravel(),
                pred=(8 + rng.normal(0, .5, len(C) * npx)).astype(np.float32), target=target.astype(np.float32),
                gt=rng.random(n_fs * npx) > 0.1, emb=emb,
                emb_index=pd.DataFrame(dict(case_id=C.case_id, row_start=np.arange(len(C)) * npx, n_pixels=npx)),
                geometry={c: nb for c in C.case_id}, feats=feats,
                groups=retrieval.prefix_groups(feats.columns, {'a': ('q_a__',), 'b': ('q_b__',)}),
                named=rng.normal(size=(len(C), 3)), daily=daily, fs_year=fs_year, fs_first_day=(fs_year - 1970) * 365 - 320,
                lag=environment.lagged(daily, C.fs.to_numpy(), np.full(len(C), 280)),
                means=rng.normal(size=(len(C), 3, dim)).astype(np.float32))


def pixels_of(D, cases):
    """Pixel ids of the field-seasons of the given cases (boolean mask over the case table)."""
    return np.unique(D['pix'][np.isin(D['row'], np.flatnonzero(cases))])


def fit(D, target, gt):
    """Steps 2-3 of fieldwise.tutorial.run with the given labels; the outputs that serve the target year."""
    C, cut = D['cases'], D['cutoff']
    B = fb.Bank(C[['case_id', 'year', 'field', 'stage', 'decision_ms', 'label_ms', 'pix0', 'n_service']].copy(),
                D['row'], D['pix'], D['pred'], target[D['pix']], gt[D['pix']])
    M = B.moments()
    B.cases['rbar'] = M.rbar.where(M.n > 0).to_numpy()
    refs = retrieval.select_references(B.cases, D['feats'], D['groups'], k=5)
    with mock.patch.multiple(rpt, **FAST_RPT):
        R = rpt.RPT(B, D['emb'], D['emb_index'], refs, D['geometry'])
        sums, s15, models = [], {}, {}
        for T in sorted(set(C.year.to_numpy()[R.queries])):
            models[T], _, sm, px = rpt.run_year(R, T, cutoff=cut)
            sums.append(sm)
            s15.update(px)
    SM = pd.concat(sums, ignore_index=True).astype(float).set_index('row').reindex(range(len(C)))
    m6 = np.column_stack([M.n, M.sr, M.srr] + [SM[k].fillna(0).to_numpy() for k in ('sS', 'srS', 'sSS')]).astype(float)
    X = C.join(M[['rbar', 'mse', 'n', 'sy', 'sp', 'spp', 'syp', 'p_mean', 'p_sd', 'log_n']])
    X.loc[X.n == 0, ['rbar', 'mse']] = np.nan
    X['w'] = selection.field_season_weights(X)
    G, env, _ = environment.fit_environment(X, D['named'], D['lag'], D['daily'], D['fs_year'], D['fs_first_day'], cutoff=cut)
    ok, years = (X.n > 0).to_numpy(), sorted(int(v) for v in C.year.unique())
    lab = X[ok]
    coef = calibration.fit_affine(lab, cutoff=cut, targets=years)
    Cal = calibration.calibration_offset(coef, X.year, X.stage, X.p_mean)
    b7 = readout.b7_features(X.p_mean.to_numpy(), X.p_sd.to_numpy(), X.log_n.to_numpy(), G, Cal)
    rhat, _ = readout.readouts(X, b7, D['means'], np.full(len(X), 2), X.rbar.to_numpy(), cutoff=cut)
    choice, _ = strength.select_strengths(lab, m6[ok], G[ok], Cal[ok], cutoff=cut, targets=years)
    gamma, beta, kappa = (X.year.map({T: v[i] for T, v in choice.items()}).to_numpy(float) for i in range(3))
    U, c = decision.offsets(rhat, G, Cal, gamma, kappa)
    with np.errstate(all='ignore'):
        L = decision.alpha_losses(m6, U, c, beta)
    alpha = decision.choose_alpha(X, L, ok, cutoff=cut, targets=years)
    q = np.flatnonzero(X.year == TARGET)
    u = U[q, alpha[TARGET]]
    pred = np.concatenate([B.pred[B.row == i] + beta[i] * s15[i] + u[j] for j, i in enumerate(q)])
    return dict(references=refs[refs.query_row.isin(q)].reset_index(drop=True), A=models[TARGET].A, tau=models[TARGET].tau,
                S15=np.concatenate([s15[i] for i in q]), G=G[q], G_pair=env[TARGET].pair, Cal=Cal[q],
                calibration=coef[coef.year == TARGET].reset_index(drop=True), rhat=rhat[q], strengths=choice[TARGET],
                alpha=alpha[TARGET], prediction=pred)


class TestLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)                                            # tiny tensors: one thread is fastest
        cls.D = study(np.random.default_rng(10))
        cls.ref = fit(cls.D, cls.D['target'], cls.D['gt'])

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def assert_unchanged(self, out):
        for k, v in self.ref.items():
            with self.subTest(output=k):
                if isinstance(v, pd.DataFrame):
                    pd.testing.assert_frame_equal(out[k], v)
                elif isinstance(v, np.ndarray):
                    np.testing.assert_array_equal(out[k], v)
                else:
                    self.assertEqual(out[k], v)

    def perturbed(self, cases, remove=False):
        """Labels with those of the given cases' field-seasons shifted, or removed."""
        target, gt = self.D['target'].copy(), self.D['gt'].copy()
        p = pixels_of(self.D, cases)
        if remove:
            target[p], gt[p] = np.nan, False
        else:
            target[p] += np.random.default_rng(11).normal(3, 2, len(p)).astype(np.float32)
        return target, gt

    def test_target_year_labels_changed(self):
        self.assert_unchanged(fit(self.D, *self.perturbed((self.D['cases'].year == TARGET).to_numpy())))

    def test_target_year_labels_removed(self):
        self.assert_unchanged(fit(self.D, *self.perturbed((self.D['cases'].year == TARGET).to_numpy(), remove=True)))

    def test_labels_after_the_cutoff_changed(self):
        self.assertTrue((self.D['cases'].label_ms[self.D['late']] > self.D['cutoff'][TARGET]).all())
        self.assert_unchanged(fit(self.D, *self.perturbed(self.D['late'])))

    def test_available_labels_change_the_target_year(self):
        """Control: the same change to a field-season whose labels are available does reach the target year."""
        C = self.D['cases']
        out = fit(self.D, *self.perturbed(((C.year == TARGET - 1) & (C.field == 1)).to_numpy()))
        self.assertFalse(np.array_equal(out['prediction'], self.ref['prediction']))


if __name__ == '__main__':
    unittest.main()
