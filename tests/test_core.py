"""Unit tests of the correction components on small synthetic data."""
import unittest

import numpy as np
import pandas as pd

from fieldwise import bank, decision, environment, retrieval, selection, strength
from fieldwise.linalg import ridge_path, ridge_two_block


def synthetic_cases(rng, years=(2018, 2019, 2020, 2021), fields=12, stages=('tau1', 'tau2')):
    rows = []
    for y in years:
        for f in range(fields):
            for k, s in enumerate(stages):
                dec = (y - 1970) * 365 * 86_400_000 + k * 30 * 86_400_000
                rows.append(dict(case_id=f'{f}:{y}:{s}', year=y, field=f, stage=s, decision_ms=dec,
                                 label_ms=dec + 200 * 86_400_000))
    C = pd.DataFrame(rows)
    C['w'] = selection.field_season_weights(C)
    return C


class TestSelection(unittest.TestCase):
    def test_field_season_weights(self):
        C = synthetic_cases(np.random.default_rng(0))
        np.testing.assert_allclose(C.groupby(['year', 'field']).w.sum(), 1.0)

    def test_first_min_prefers_simpler(self):
        self.assertEqual(selection.first_min([1.0, 1.0 + 1e-13, 0.5 + 1.0]), 0)

    def test_forward_choice_uses_only_earlier_years(self):
        C = synthetic_cases(np.random.default_rng(0))
        L = np.zeros((len(C), 2))
        L[:, 1] = np.where(C.year == 2021, -1.0, 1.0)            # candidate 1 looks good only in the target year itself
        pick = selection.forward_choice(C, L, np.ones(len(C), bool), [2020, 2021])
        self.assertEqual(pick[2021][0], 0)


class TestBank(unittest.TestCase):
    def test_moments(self):
        rng = np.random.default_rng(1)
        row = np.repeat(np.arange(3), 50)
        pred, target = rng.normal(5, 1, 150).astype(np.float32), rng.normal(6, 1, 150).astype(np.float32)
        gt = rng.random(150) > 0.2
        b = bank.Bank(pd.DataFrame(dict(case_id=['a', 'b', 'c'])), row, np.arange(150), pred, target, gt)
        M = b.moments()
        for i in range(3):
            m = (row == i) & gt
            r = target[m].astype(np.float64) - pred[m]
            self.assertAlmostEqual(M.rbar[i], r.mean(), places=10)
            self.assertAlmostEqual(M.mse[i], (r ** 2).mean(), places=9)


class TestRetrieval(unittest.TestCase):
    def test_eligibility_and_weights(self):
        rng = np.random.default_rng(2)
        C = synthetic_cases(rng)
        feats = pd.DataFrame(dict(q_a__x=rng.normal(size=len(C)), q_b__y=rng.normal(size=len(C))))
        refs = retrieval.select_references(C, feats, retrieval.prefix_groups(feats.columns, {'a': ('q_a__',), 'b': ('q_b__',)}), k=5)
        q, d = C.iloc[refs.query_row].reset_index(drop=True), C.iloc[refs.donor_row].reset_index(drop=True)
        self.assertTrue((d.year < q.year).all() and (d.stage == q.stage).all())
        self.assertTrue((d.label_ms <= q.decision_ms).all())
        np.testing.assert_allclose(refs.groupby('query_row').weight.sum(), 1.0)


class TestRidge(unittest.TestCase):
    def test_two_block_matches_single_penalty(self):
        rng = np.random.default_rng(3)
        X, y, w = rng.normal(size=(40, 5)), rng.normal(size=40), rng.random(40) + 0.5
        p1 = ridge_path(X, y, w, X, [3.0])[0][:, 0]
        p2 = ridge_two_block(X, y, w, X, 2, [(3.0, 3.0)])[0][:, 0]
        np.testing.assert_allclose(p1, p2, atol=1e-10)

    def test_environment_year_model_reproduces_fit(self):
        rng = np.random.default_rng(4)
        C = synthetic_cases(rng)
        C['rbar'] = rng.normal(size=len(C))
        C['mse'] = C.rbar ** 2 + 1
        named = rng.normal(size=(len(C), 3))
        daily = rng.normal(size=(C.groupby(['year', 'field']).ngroups, 300, 2))
        fs_row = C.groupby(['year', 'field']).ngroup().to_numpy()
        fs_year = C.drop_duplicates(['year', 'field']).sort_values(['year', 'field']).year.to_numpy()
        lag = environment.lagged(daily, fs_row, np.full(len(C), 280))
        first = (fs_year - 1970) * 365 - 320                      # series end 20 days before the year's first decision
        G, models, _ = environment.fit_environment(C, named, lag, daily, fs_year, first)
        q = (C.year == 2021).to_numpy()
        np.testing.assert_allclose(models[2021].predict(named[q], lag[q]), G[q], atol=1e-12)


class TestTiming(unittest.TestCase):
    """Legal history: decision and labels by the cutoff; inputs available after the cutoff are left out."""

    def test_legal_mask_checks_decision_and_label(self):
        C = synthetic_cases(np.random.default_rng(6))
        cut = {2021: float(C[C.year == 2021].decision_ms.min())}
        late = (C.year == 2020) & (C.stage == 'tau2')
        C.loc[late, 'decision_ms'] = cut[2021] + 1                 # decided after the cutoff, label before it
        C.loc[late, 'label_ms'] = cut[2021] - 1
        m = selection.legal_mask(C, 2021, cut)
        self.assertFalse(m[late.to_numpy()].any())
        self.assertTrue(m[((C.year == 2020) & (C.stage == 'tau1')).to_numpy()].all())

    def test_daily_scaler_uses_days_available_by_cutoff(self):
        daily = np.zeros((2, 10, 1))
        daily[0, 5:] = 100.0                                       # values that become available late
        avail = environment.day_available_ms(np.array([0, 0]), 10)
        cut = float(avail[0, 4])                                   # days 0..4 available
        mu, sd = environment.daily_scaler(daily, np.array([2000, 2000]), 2001, avail, cut)
        self.assertEqual(mu[0], 0.0)
        self.assertEqual(sd[0], 0.0)

    def test_new_year_without_labels(self):
        rng = np.random.default_rng(7)
        C = synthetic_cases(rng, years=(2018, 2019, 2020, 2021, 2022))
        new = (C.year == 2022).to_numpy()
        C['rbar'] = np.where(new, np.nan, rng.normal(size=len(C)))
        C['mse'] = C.rbar ** 2 + 1
        cut = selection.earliest_decision(C).to_dict()
        named = rng.normal(size=(len(C), 3))
        daily = rng.normal(size=(C.groupby(['year', 'field']).ngroups, 300, 2))
        fs_row = C.groupby(['year', 'field']).ngroup().to_numpy()
        fs_year = C.drop_duplicates(['year', 'field']).sort_values(['year', 'field']).year.to_numpy()
        lag = environment.lagged(daily, fs_row, np.full(len(C), 280))
        G, models, _ = environment.fit_environment(C, named, lag, daily, fs_year, (fs_year - 1970) * 365 - 320, cutoff=cut)
        self.assertTrue(np.isfinite(G[new]).all() and 2022 in models)
        # the new year's queries do not change any fit when their (absent) labels are replaced
        C2 = C.copy()
        C2.loc[new, 'rbar'] = 5.0
        G2, _, _ = environment.fit_environment(C2, named, lag, daily, fs_year, (fs_year - 1970) * 365 - 320, cutoff=cut)
        np.testing.assert_array_equal(G[new], G2[new])
        from fieldwise import readout
        means = rng.normal(size=(len(C), 3, 4)).astype(np.float32)
        b7 = rng.normal(size=(len(C), 7))
        rhat, models_r = readout.readouts(C, b7, means, np.full(len(C), 2), C.rbar.to_numpy(), cutoff=cut)
        self.assertTrue(np.isfinite(rhat[new]).all() and ('tau1', 2022) in models_r)
        lab = C[~new].assign(sy=1.0, sp=1.0, spp=1.5, syp=1.2)
        from fieldwise import calibration
        coef = calibration.fit_affine(lab, cutoff=cut, targets=[2022])
        self.assertEqual(sorted(coef.stage), ['tau1', 'tau2'])

    def test_one_crop_per_call(self):
        C = synthetic_cases(np.random.default_rng(9))
        mixed = C.assign(crop=np.where(C.field < 6, 'maize', 'soybean'))
        feats = pd.DataFrame(dict(q_a__x=np.zeros(len(C))))
        for call in (lambda: selection.cutoff_table(mixed), lambda: selection.legal_mask(mixed, 2021),
                     lambda: retrieval.select_references(mixed, feats, {'a': ['q_a__x']})):
            with self.assertRaises(ValueError):
                call()
        np.testing.assert_array_equal(selection.legal_mask(C.assign(crop='maize'), 2021), selection.legal_mask(C, 2021))

    def test_retrieval_standardises_on_inputs_available_to_all_queries(self):
        rng = np.random.default_rng(8)
        C = synthetic_cases(rng, years=(2019, 2020), fields=6, stages=('tau1',))
        feats = pd.DataFrame(dict(q_a__x=rng.normal(size=len(C))))
        groups = retrieval.prefix_groups(feats.columns, {'a': ('q_a__',)})
        late = C.index[(C.year == 2019)][:1]
        C.loc[late, 'decision_ms'] = C[C.year == 2020].decision_ms.min() + 1    # decided after every 2020 query
        r1 = retrieval.select_references(C, feats, groups, k=10)
        feats2 = feats.copy()
        feats2.loc[late, 'q_a__x'] = 1e6                                         # its features cannot matter
        r2 = retrieval.select_references(C, feats2, groups, k=10)
        pd.testing.assert_frame_equal(r1, r2)


class TestDecision(unittest.TestCase):
    def test_shrink_bounds(self):
        c = np.array([1.0, -1.0, 0.0, 2.0])
        u = decision.shrink(np.array([0.5, 3.0, 1.0, -1.0]), c)
        np.testing.assert_allclose(u, [0.5, -0.0, 0.0, 0.0])

    def test_f16_picks_closest_candidate(self):
        G, Cal = np.array([2.0]), np.array([0.0])
        u = decision.f16(np.array([1.1]), G, Cal, decision.candidate_order(np.array([0.5]), np.array([0.5])))
        self.assertAlmostEqual(float(u[0]), 1.0)

    def test_shift_loss_exact(self):
        rng = np.random.default_rng(5)
        r, S = rng.normal(size=100), rng.normal(size=100)
        m = np.array([100, r.sum(), (r * r).sum(), S.sum(), (r * S).sum(), (S * S).sum()])
        u, beta = 0.3, 0.5
        self.assertAlmostEqual(float(strength.shift_loss(m, u, beta)), float(((r - u - beta * S) ** 2).mean()), places=10)


if __name__ == '__main__':
    unittest.main()
