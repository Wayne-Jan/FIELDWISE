"""End-to-end tests of the direct pixel-level entry point (fieldwise.correct_pixels)."""
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

import fieldwise
from fieldwise import rpt

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_pixel_dataset(seed=0, years=(2020, 2021, 2022, 2023), fields_per_year=6, pixels_per_field=25, dim=6):
    """Synthetic multi-year pixel-level dataset with a learnable within-field residual pattern and a field offset."""
    rng = np.random.default_rng(seed)
    rows, embs = [], []
    side = int(round(np.sqrt(pixels_per_field)))
    gx, gy = np.meshgrid(np.arange(side) * 10.0, np.arange(side) * 10.0, indexing='ij')
    gx, gy = gx.ravel()[:pixels_per_field], gy.ravel()[:pixels_per_field]
    true_w = np.array([1.2, -0.8, 0.5, 0.0, 0.0, 0.0])[:dim]
    for y in years:
        for f in range(fields_per_year):
            cid = f'{y}:f{f}:tau3'
            e = rng.normal(0.0, 1.0, (pixels_per_field, dim)).astype(np.float32)
            # Smooth spatial structure + pixel-level residual tied to embedding direction
            pattern = (e @ true_w) - (e @ true_w).mean()
            base_pred = 7.0 + rng.normal(0.0, 0.4, pixels_per_field)
            target = base_pred + 0.4 + pattern + rng.normal(0.0, 0.2, pixels_per_field)
            for p in range(pixels_per_field):
                rows.append(dict(
                    case_id=cid, year=y, field=f'f{f}', stage='tau3',
                    x=float(gx[p]), y=float(gy[p]),
                    pred=float(base_pred[p]), target=float(target[p]),
                ))
            embs.append(e)
    df = pd.DataFrame(rows)
    E = np.vstack(embs)
    # Shuffle row order to verify stable round-trip alignment
    perm = rng.permutation(len(df))
    return df.iloc[perm].reset_index(drop=True), E[perm]


class TestPixelPipeline(unittest.TestCase):
    def test_correct_pixels_improves_within_field_mse_and_preserves_row_order(self):
        df, emb = make_pixel_dataset(seed=0)
        # Keep unit test fast on CPU-only machines while running full RPT fitting on GPU when available
        patches = () if DEV == 'cuda' else (
            mock.patch.object(rpt, 'QS', 32), mock.patch.object(rpt, 'MS', 32),
            mock.patch.object(rpt, 'VAL_QS', 32), mock.patch.object(rpt, 'VAL_MS', 32),
            mock.patch.object(rpt, 'MAX_EPOCHS', 6),
        )
        for p in patches:
            p.start()
        try:
            res = fieldwise.correct_pixels(df, emb, k=10, seed=0, device=DEV)
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(len(res.pixels), len(df))
        np.testing.assert_array_equal(res.pixels['case_id'].to_numpy(), df['case_id'].to_numpy())
        np.testing.assert_allclose(res.pixels['pred_base'].to_numpy(), df['pred'].to_numpy(), atol=1e-6)
        # Final target year (2023) should have finite corrections and lower MSE than Base
        t = res.pixels[res.pixels['year'] == 2023]
        self.assertTrue(np.isfinite(t['pred_fieldwise']).all())
        mse_base = float(np.mean((t['target'] - t['pred_base']) ** 2))
        mse_fw = float(np.mean((t['target'] - t['pred_fieldwise']) ** 2))
        self.assertLess(mse_fw, mse_base)

    def test_strict_temporal_legality_target_year_labels_never_leak(self):
        df, emb = make_pixel_dataset(seed=3)
        df_unlabelled = df.copy()
        df_unlabelled.loc[df_unlabelled['year'] == 2023, 'target'] = np.nan
        patches = () if DEV == 'cuda' else (
            mock.patch.object(rpt, 'QS', 32), mock.patch.object(rpt, 'MS', 32),
            mock.patch.object(rpt, 'VAL_QS', 32), mock.patch.object(rpt, 'VAL_MS', 32),
            mock.patch.object(rpt, 'MAX_EPOCHS', 4),
        )
        for p in patches:
            p.start()
        try:
            res_a = fieldwise.correct_pixels(df, emb, k=10, seed=1, device=DEV)
            res_b = fieldwise.correct_pixels(df_unlabelled, emb, k=10, seed=1, device=DEV)
        finally:
            for p in patches:
                p.stop()
        m = df['year'].to_numpy() == 2023
        np.testing.assert_allclose(res_a.pixels.loc[m, 's15'], res_b.pixels.loc[m, 's15'], atol=0.0)
        np.testing.assert_allclose(res_a.pixels.loc[m, 'pred_fieldwise'], res_b.pixels.loc[m, 'pred_fieldwise'], atol=0.0)


if __name__ == '__main__':
    unittest.main()
