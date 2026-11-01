"""Edge cases of the decision schedule (fieldwise.schedule) on small synthetic temperature series."""
import unittest

import numpy as np
import pandas as pd

from fieldwise import schedule as sc

DAY = sc.DAY_MS


def day(s):
    return int((pd.Timestamp(s) - pd.Timestamp('1970-01-01')).days)


def seasons(n, crop='corn', sowing=None, harvest=None, sowing_type='provided_by_farmer'):
    return pd.DataFrame(dict(season_id=np.arange(n), crop=crop, sowing_day=sowing, harvest_day=harvest, sowing_type=sowing_type))


class TestDailyMean(unittest.TestCase):
    def test_missing_hour_invalidates_utc_day(self):
        h = np.full((3, 24), 300.0)
        h[1, 23] = np.nan          # last hour of the second UTC day
        m = sc.daily_mean(h)
        self.assertTrue(np.isfinite(m[0]) and np.isnan(m[1]) and np.isfinite(m[2]))

    def test_mean_of_24_hours(self):
        h = np.arange(48, dtype=float).reshape(2, 24)
        np.testing.assert_allclose(sc.daily_mean(h), [11.5, 35.5])


class TestThresholds(unittest.TestCase):
    def test_equality_is_a_crossing(self):
        t = np.full((1, 10), 20.0)                      # 10 GDD per day at base 10
        d = sc.threshold_days(t, [100], [200], [100], 10.0, (40.0,))
        self.assertEqual(d[0, 0], 103)                  # cumulative 40 on the fourth day

    def test_gap_blocks_later_crossing(self):
        t = np.full((1, 10), 20.0)
        t[0, 2] = np.nan
        d = sc.threshold_days(t, [100], [200], [100], 10.0, (20.0, 40.0))
        self.assertEqual(d[0, 0], 101)                  # reached before the gap
        self.assertTrue(np.isnan(d[0, 1]))              # crossing after an invalid day is unknown

    def test_not_reached_before_harvest(self):
        t = np.full((1, 30), 20.0)
        d = sc.threshold_days(t, [100], [104], [100], 10.0, (40.0, 60.0))
        self.assertEqual(d[0, 0], 103)
        self.assertTrue(np.isnan(d[0, 1]))              # days after harvest do not count

    def test_days_before_sowing_ignored(self):
        t = np.full((1, 10), 20.0)
        d = sc.threshold_days(t, [103], [200], [100], 10.0, (20.0,))
        self.assertEqual(d[0, 0], 104)

    def test_cold_days_add_nothing(self):
        t = np.array([[5.0, 30.0, 5.0, 30.0]])
        d = sc.threshold_days(t, [0], [10], [0], 10.0, (40.0,))
        self.assertEqual(d[0, 0], 3)


class TestSchedule(unittest.TestCase):
    def run_one(self, sow, harv, temp=20.0, crop='corn', sowing_type='provided_by_farmer', n=None):
        n = n or (harv - sow + 1)
        S = seasons(1, crop, [sow], [harv], sowing_type)
        return sc.schedule(S, np.full((1, n), temp), np.array([sow])).set_index('stage')

    def test_window_decision_label_times(self):
        sow, harv = day('2023-10-02'), day('2024-04-04')
        T = self.run_one(sow, harv)                     # 10 GDD per day: tau1 = 400 GDD on day 40
        self.assertEqual(T.loc['tau1', 'threshold_day'], sow + 39)
        self.assertEqual(T.loc['tau1', 'window_end_ms'], (sow + 40) * DAY)
        self.assertEqual(T.loc['tau1', 'decision_ms'], (sow + 46) * DAY)
        self.assertEqual(T.loc['tau1', 'label_ms'], day('2024-05-04') * DAY)
        self.assertEqual(T.loc['harvest', 'window_end_ms'], (harv + 1) * DAY)
        self.assertEqual(T.loc['harvest', 'decision_ms'], (harv + 7) * DAY)
        self.assertTrue(T.eligible.all())

    def test_season_across_new_year(self):
        sow = day('2023-12-20')
        T = self.run_one(sow, day('2024-06-01'))
        self.assertEqual(pd.Timestamp(int(T.loc['tau1', 'window_end_ms']), unit='ms'), pd.Timestamp('2024-01-29'))

    def test_decision_on_harvest_date_excluded(self):
        sow = day('2024-01-01')
        harv = sow + 39 + 1 + 6                         # tau1 decision lands exactly on the harvest date
        T = self.run_one(sow, harv)
        self.assertEqual(T.loc['tau1', 'decision_ms'], harv * DAY)
        self.assertEqual(T.loc['tau1', 'reason'], 'decision_on_or_after_harvest')
        T2 = self.run_one(sow, harv + 1)                # one day later the decision precedes harvest
        self.assertTrue(T2.loc['tau1', 'eligible'])

    def test_unknown_threshold_excluded(self):
        sow = day('2024-01-01')
        T = self.run_one(sow, sow + 100)                # 1010 GDD: tau3, tau4 not reached
        self.assertEqual(list(T.reason[['tau3', 'tau4']]), ['threshold_unknown'] * 2)

    def test_sowing_without_clock_drops_season(self):
        sow = day('2024-01-01')
        T = self.run_one(sow, sow + 300, sowing_type='290 days before harvest')
        self.assertFalse(T.eligible.any())

    def test_no_eligible_decision_drops_harvest_case(self):
        sow = day('2024-01-01')
        T = self.run_one(sow, sow + 300, temp=10.0)     # no GDD at all
        self.assertEqual(T.loc['harvest', 'reason'], 'no_eligible_decision')

    def test_crop_thresholds(self):
        sow = day('2023-10-15')
        T = self.run_one(sow, sow + 400, temp=10.0, crop='wheat')     # base 0: 10 GDD per day, 600 on day 60
        self.assertEqual(T.loc['tau1', 'threshold_day'], sow + 59)


if __name__ == '__main__':
    unittest.main()
