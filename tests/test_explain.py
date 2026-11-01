"""Unit tests of the explanation tools on small synthetic data (they also show how to call them)."""
import unittest

import numpy as np
import pandas as pd

from fieldwise import evaluation, readout
from fieldwise.decision import candidate_order
from fieldwise.explain import decisions, erasure, errors, increments, spatial
from fieldwise.strength import shift_loss


def synthetic_errors(rng, years=(2020, 2021, 2022), fields=30, stages=('tau1', 'tau2'), pixels=40):
    """Cases with per-pixel residuals r and a within-field pattern S, as per-case sums (strength.MOMENTS)."""
    rows, m = [], []
    for y in years:
        for f in range(fields):
            country = 'A' if f < fields // 2 else 'B'
            for s in stages:
                r = rng.normal(0.3 * (y - 2021) + (0.5 if country == 'A' else -0.5) + rng.normal(0, .4), 1.0, pixels)
                S = 0.5 * (r - r.mean()) + rng.normal(0, .5, pixels)
                rows.append(dict(year=y, stage=s, country=country, fid=y * 1000 + f))
                m.append([pixels, r.sum(), (r * r).sum(), S.sum(), (r * S).sum(), (S * S).sum()])
    C = pd.DataFrame(rows)
    C['w'] = 1.0 / len(stages)
    return C, np.array(m)


class TestErrors(unittest.TestCase):
    def test_layers_add_up_to_mse(self):
        C, m = synthetic_errors(np.random.default_rng(0))
        L = errors.Layers(C.year, C.stage, C.w, dict(country=C.country))
        x, mse = m[:, 1] / m[:, 0], m[:, 2] / m[:, 0]
        sizes = L.sizes(x, mse)[0]
        self.assertEqual(L.names, ['year_stage', 'country', 'field', 'within'])
        self.assertAlmostEqual(sizes.sum(), evaluation.scalar_rollup(C.year.to_numpy(), C.w.to_numpy(), mse[:, None])[0], places=10)
        self.assertTrue((sizes >= -1e-12).all())

    def test_shapley_adds_up_in_every_draw(self):
        rng = np.random.default_rng(1)
        C, m = synthetic_errors(rng)
        G, Cal = rng.normal(0, .5, len(C)), rng.normal(0, .3, len(C))
        draws = evaluation.draw_matrix(C.drop_duplicates('fid').set_index('fid').year.sort_index(), rng, draws=50)
        L = errors.Layers(C.year, C.stage, C.w, dict(country=C.country), C.fid, draws)
        one = np.ones(len(C))
        phi, E = errors.correction_contributions(L, m, G, Cal, 0.5 * one, one, 0.5 * one)
        total = 100 * np.nanmean(E[frozenset()] - E[frozenset({'RPT', 'G', 'K'})], 1) / np.nanmean(E[frozenset()].sum(2), 1)[:, None]
        np.testing.assert_allclose(sum(phi.values()), total, atol=1e-10)


class TestSpatial(unittest.TestCase):
    def test_boundary_distance_of_a_square(self):
        rr, cc = np.meshgrid(np.arange(9), np.arange(9), indexing='ij')
        d = spatial.boundary_distance(rr.ravel(), cc.ravel())
        self.assertAlmostEqual(d.reshape(9, 9)[4, 4], 45.0)
        self.assertAlmostEqual(d.reshape(9, 9)[0, 4], 5.0)

    def test_contrast_and_slope(self):
        rr, cc = np.meshgrid(np.arange(9), np.arange(9), indexing='ij')
        d = spatial.boundary_distance(rr.ravel(), cc.ravel())
        y = np.where(d < 20, 8.0, 10.0)
        case = np.zeros(len(d), int)
        ok = np.ones(len(d), bool)
        self.assertAlmostEqual(spatial.edge_contrast(y, d, ok, case, 1)[0], -2.0)
        self.assertAlmostEqual(spatial.response_slope(y, 2 * y + 1, ok, case, 1)[0], 2.0)


class TestDecisions(unittest.TestCase):
    def test_loss_change_matches_shift_loss(self):
        rng = np.random.default_rng(2)
        _, m = synthetic_errors(rng)
        beta, c, u = np.full(len(m), .5), rng.normal(0, 1, len(m)), rng.normal(0, 1, len(m))
        e = (m[:, 1] - beta * m[:, 3]) / m[:, 0] - c
        delta, cats = decisions.loss_change(u - c, e)
        np.testing.assert_allclose(delta, shift_loss(m, u, beta) - shift_loss(m, c, beta), atol=1e-9)
        self.assertTrue((sum(v.astype(int) for v in cats.values()) == 1).all())

    def test_fractions_end_at_correction_only(self):
        rng = np.random.default_rng(3)
        _, m = synthetic_errors(rng)
        beta, c = np.ones(len(m)), rng.normal(0, 1, len(m))
        L = decisions.fraction_losses(m, c, beta)
        np.testing.assert_allclose(L[:, -1], shift_loss(m, c, beta), atol=1e-12)
        np.testing.assert_allclose(L[:, 0], shift_loss(m, 0 * c, beta), atol=1e-12)

    def test_permutation_stays_in_group(self):
        C, _ = synthetic_errors(np.random.default_rng(4))
        C['case_id'] = C.fid.astype(str) + ':' + C.stage
        rhat = np.arange(len(C), dtype=float)
        groups = decisions.permutation_groups(C, np.ones(len(C), bool), ['year', 'stage'])
        R = decisions.permuted_readouts(rhat, groups, np.random.default_rng(0), n_perm=5)
        for ix in groups:
            self.assertTrue((np.sort(R[:, ix], 1) == np.sort(rhat[ix])).all())

    def test_layer_offsets_without_readout_keep_c(self):
        G, Cal = np.array([1.0, 2.0]), np.array([0.5, -1.0])
        c = 0.5 * G + 0.5 * Cal
        order = candidate_order(np.full(2, .5), np.full(2, .5))
        uS, uF = decisions.layer_offsets(np.array([np.nan, 10.0]), G, Cal, c, order, np.full(2, .5), np.array([False, True]))
        self.assertEqual(uS[0], c[0])
        self.assertEqual(uF[0], c[0])
        self.assertAlmostEqual(uS[1], c[1])                     # readout above c: shrinkage keeps c


class TestErasure(unittest.TestCase):
    def test_leace_removes_linear_association(self):
        rng = np.random.default_rng(5)
        n = 300
        Z = rng.normal(size=(n, 2))
        X = np.column_stack([Z @ rng.normal(size=(2, 6)), rng.normal(size=(n, 4))]) + rng.normal(0, .1, (n, 10))
        year = np.repeat([2020, 2021, 2022], n // 3)
        fit = np.arange(n)
        er, _ = erasure.fit_erasers(X, Z, fit, year, seeds=(1, 2))
        Xe = er['named'](X)
        w = erasure.year_weights(year)
        z = erasure.standardize(Z, fit)
        cross = (Xe - w @ Xe).T @ (w[:, None] * (z - w @ z))
        self.assertLess(np.abs(cross).max(), 1e-8)
        for k in ('random1', 'random2'):
            self.assertEqual(er[k].rank, er['named'].rank)
            self.assertAlmostEqual(np.trace(er[k].L @ er[k].L.T), er['named'].target, places=6)


class TestIncrements(unittest.TestCase):
    def test_two_blocks_match_the_readout(self):
        rng = np.random.default_rng(6)
        n = 120
        year = np.repeat([2019, 2020, 2021, 2022], n // 4)
        XB, XF = rng.normal(size=(n, 7)), rng.normal(size=(n, 12))
        y = XB[:, 0] + 0.5 * XF[:, 1] + rng.normal(0, .3, n)
        train, query = np.flatnonzero(year < 2022), np.flatnonzero(year == 2022)
        rh, lam = increments.fit_blocks([XB, XF], y, year, train, query)
        model = readout.fit_readout(XB, XF, y, year, train)
        np.testing.assert_allclose(rh, model.predict(XB[query], XF[query]), atol=1e-10)
        self.assertEqual(tuple(lam), model.lambdas)

    def test_increment_split(self):
        L_S, L_F, L_N, L_FN, M = (np.array([v]) for v in (5., 4., 4.5, 3.8, 10.))
        I = increments.increments(L_S, L_F, L_N, L_FN, M)
        self.assertAlmostEqual((I['shared'] + I['F_given_N'])[0], I['I_F'][0])
        self.assertAlmostEqual((I['shared'] + I['N_given_F'])[0], I['I_N'][0])


if __name__ == '__main__':
    unittest.main()
