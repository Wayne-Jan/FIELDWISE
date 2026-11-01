"""Checks of prepared inputs (fieldwise.validate) on small synthetic tables."""
import unittest

import numpy as np
import pandas as pd

from fieldwise import bank, selection, validate

DAY = 86_400_000


def table():
    rows = []
    for y in (2019, 2020, 2021):
        for f in range(3):
            for k, s in enumerate(('tau1', 'tau2', 'harvest')):
                dec = ((y - 1970) * 365 + 30 * k + f) * DAY
                rows.append(dict(case_id=f'{f}:{y}:{s}', year=y, field=f, stage=s, decision_ms=dec,
                                 window_end_ms=dec - 6 * DAY, label_ms=((y - 1970) * 365 + 120) * DAY))
    return pd.DataFrame(rows)


class TestValidate(unittest.TestCase):
    def test_clean_inputs_pass(self):
        C = table()
        cut = selection.cutoff_table(C)
        self.assertEqual(len(validate.cases(C)), 0)
        self.assertEqual(len(validate.cutoffs(cut, C)), 0)
        self.assertEqual(validate.assert_valid(validate.cases(C), validate.cutoffs(cut, C)).shape[0], 0)

    def test_case_errors(self):
        C = table()
        C.loc[0, 'label_ms'] = C.loc[0, 'decision_ms']                  # label available at the decision
        C.loc[1, 'window_end_ms'] = C.loc[1, 'decision_ms'] + 1         # window after the decision
        C.loc[2, 'case_id'] = C.loc[3, 'case_id']
        found = set(validate.cases(C).check)
        self.assertTrue({'decision_before_label', 'window_end_ms_by_decision', 'case_id_unique'} <= found)
        with self.assertRaises(ValueError):
            validate.assert_valid(validate.cases(C))

    def test_cutoff_after_first_query_is_an_error(self):
        C = table()
        cut = selection.cutoff_table(C)
        cut.loc[0, 'cutoff_ms'] += DAY
        cut = pd.concat([cut, pd.DataFrame(dict(year=[2022], phase=['in_season'], cutoff_ms=[0.0]))])
        I = validate.cutoffs(cut.iloc[1:], C)                          # first row dropped: a missing cutoff
        self.assertIn('cutoff_present', set(I.check))
        self.assertIn('cutoff_not_after_queries', set(validate.cutoffs(cut, C).check))

    def test_mixed_crops_rejected(self):
        C = table()
        mixed = C.assign(crop=np.where(C.field == 0, 'maize', 'soybean'))
        self.assertIn('one_crop', set(validate.cases(mixed).check))
        self.assertEqual(len(validate.cases(C.assign(crop='maize'))), 0)
        cut = selection.cutoff_table(C)
        self.assertIn('one_crop', set(validate.cutoffs(cut, mixed).check))
        self.assertIn('one_crop', set(validate.cutoffs(cut.assign(crop='wheat'), C.assign(crop='maize')).check))
        with self.assertRaises(ValueError):
            validate.late_history(mixed, cut)

    def test_phases_share_cutoffs(self):
        C = table()
        cut = selection.cutoff_table(C)
        self.assertEqual(sorted(cut.phase.unique()), ['harvest', 'in_season'])
        self.assertEqual(len(cut), 6)

    def test_late_history_listed(self):
        C = table()
        C.loc[(C.year == 2020) & (C.stage == 'tau2'), 'label_ms'] = ((2021 - 1970) * 365 + 200) * DAY
        L = validate.late_history(C, selection.cutoff_table(C)).set_index(['year', 'phase'])
        self.assertEqual(L.loc[(2021, 'in_season'), 'labelled_late'], 3)

    def test_bank_and_embeddings(self):
        C = table()
        rng = np.random.default_rng(0)
        row = np.repeat(np.arange(3), 4)
        target = rng.normal(size=12)
        gt = np.ones(12, bool)
        target[0] = np.nan
        B = bank.Bank(pd.DataFrame(dict(case_id=C.case_id[:3])), row, np.arange(12), rng.normal(size=12), target, gt)
        self.assertIn('bank_label_finite', set(validate.bank(B, C).check))
        I = pd.DataFrame(dict(case_id=C.case_id[:3], row_start=[0, 4, 8], n_pixels=[4, 4, 3]))
        self.assertIn('embedding_pixels', set(validate.embeddings(I, B).check))

    def test_daily_and_intervals(self):
        C = table().iloc[:2]
        first = np.array([C.window_end_ms.iloc[0] // DAY - 10])
        self.assertEqual(len(validate.daily(C.iloc[:1], [0], [10], first)), 0)
        self.assertIn('daily_within_window', set(validate.daily(C.iloc[:1], [0], [10 + 1], first).check))
        # a decision 3 days after the window end: the last day is inside the window but not yet published (6 days)
        short = C.iloc[:1].assign(decision_ms=C.window_end_ms.iloc[0] + 3 * DAY, label_ms=C.label_ms.iloc[0])
        found = set(validate.daily(short, [0], [10], first).check)
        self.assertEqual(found, {'daily_available_by_decision'})
        self.assertEqual(len(validate.daily(short, [0], [7], first)), 0)            # data up to 3 days before the end
        self.assertEqual(len(validate.daily(short, [0], [10], first, delay_days=3)), 0)
        F = pd.DataFrame({'q_s2__stage0__ndvi': [1.0, 1.0], 'q_s2__stage1__ndvi': [1.0, 2.0]})
        self.assertIn('interval_masked', set(validate.interval_features(C, F).check))   # tau1 already has interval 1


if __name__ == '__main__':
    unittest.main()
