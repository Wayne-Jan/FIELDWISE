"""Unit tests of the Base date inputs on small synthetic cases."""
import math
import unittest
from unittest import mock

import numpy as np
import torch

from fieldwise.base import data as D
from fieldwise.base.models import batch3d, date_encoding

START = 19_400                                          # UTC day number of the field-season input start
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def season_cases(n_pix=3, seed=0):
    """Two decision times of one field-season: the same input start, the later window holding more images and days."""
    rng = np.random.default_rng(seed)
    img_day = START + np.array([2, 7, 12, 17, 22, 27])
    cases = []
    for k, (n_img, n_days) in enumerate(((4, 20), (6, 30))):
        ok = np.ones((n_pix, n_img), bool)
        ok[:, 1] = False                                # one invalid step
        cases.append(dict(case_id=f'f1:2023:{k}', pix0=0, s2=rng.uniform(500, 3000, (n_pix, n_img, D.NB)).astype(np.float32),
                          scl=np.full((n_pix, n_img), 4, np.uint8), ok=ok, day=img_day[:n_img].astype(np.int32),
                          weather=rng.normal(290, 5, (n_days, 4)).astype(np.float32),
                          weather_day=np.arange(START, START + n_days, dtype=np.int32), input_start_day=START))
    soil = rng.normal(size=(n_pix, 24)).astype(np.float32)
    dem = rng.normal(size=(n_pix, 5)).astype(np.float32)
    return D.PixelData(cases, soil, dem, rng.normal(8, 1, n_pix).astype(np.float32), np.ones(n_pix, bool), samples='all')


class TestDateEncoding(unittest.TestCase):
    def setUp(self):
        self.pos = torch.arange(0, 320, dtype=torch.float32, device=DEV).reshape(4, 80)
        self.pos[0, -5:] = D.DATE_PAD
        self.omega = 1.0 / (10000 ** (torch.arange(16, dtype=torch.float32, device=DEV) / 16.0))

    def test_bf16_autocast_gives_the_fp32_encoding(self):
        ref = date_encoding(32, self.pos, self.omega)
        with torch.autocast(device_type=DEV, dtype=torch.bfloat16):
            enc = date_encoding(32, self.pos, self.omega)
        self.assertEqual(enc.dtype, torch.float32)
        torch.testing.assert_close(enc, ref, rtol=0, atol=1e-6)
        a = self.pos.double().clamp(min=0)[..., None] * self.omega.double()
        exact = torch.cat([torch.sin(a), torch.cos(a)], -1)
        torch.testing.assert_close(enc.double(), exact, rtol=0, atol=1e-4)

    def test_frequency_products_run_in_fp32_outside_autocast(self):
        seen, einsum = [], torch.einsum

        def spy(eq, *ops):
            seen.append((torch.is_autocast_enabled(DEV), [o.dtype for o in ops]))
            return einsum(eq, *ops)
        with mock.patch.object(torch, 'einsum', spy), torch.autocast(device_type=DEV, dtype=torch.bfloat16):
            date_encoding(32, self.pos, self.omega.to(torch.bfloat16))
        self.assertEqual(seen, [(False, [torch.float32, torch.float32])])


class TestSeasonDates(unittest.TestCase):
    def test_transformer_dates_count_from_the_season_input_start(self):
        p = season_cases()
        b = D.batch(p, torch.arange(p.n), D.fit_stats(p))
        wd, sd = b['x_weather_dates'], b['x_s2_dates']
        early, late = p.s_case == 0, p.s_case == 1
        n_late = int(late.sum())
        self.assertTrue(bool((wd[:, 0] == 0).all()))                                   # first weather day is the origin
        torch.testing.assert_close(wd[late], torch.arange(30.0).expand(n_late, -1))
        self.assertTrue(bool((wd[early, 20:] == D.DATE_PAD).all()))                    # padding beyond the shorter window
        torch.testing.assert_close(sd[late], torch.tensor([2.0, D.DATE_PAD, 12, 17, 22, 27]).expand(n_late, -1))
        torch.testing.assert_close(sd[early, :4], sd[late, :4])                       # same origin at both decision times
        self.assertTrue(bool((sd[early, 4:] == D.DATE_PAD).all()))

    def test_3dlstm_date_features_use_the_utc_day(self):
        p = season_cases()
        p.nb_sample = torch.full((p.n, 24), -1, dtype=torch.int64)
        b = batch3d(p, torch.arange(p.n), D.fit_stats(p))
        late = int(torch.nonzero(p.s_case == 1)[0])
        d = START + np.array([2, 12, 17, 22, 27])                                      # the valid steps, compacted
        f = b['centre'][late, :5, -3:].double().numpy()
        np.testing.assert_allclose(f[:, 0], np.sin(2 * math.pi * d / 365.25), atol=1e-4)
        np.testing.assert_allclose(f[:, 1], np.cos(2 * math.pi * d / 365.25), atol=1e-4)
        np.testing.assert_allclose(f[:, 2], (d - d[0]) / 365.0, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
