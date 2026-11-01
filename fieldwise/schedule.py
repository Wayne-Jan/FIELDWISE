"""Decision times, observation windows and label availability (manuscript Section 3.3.1, Appendix A.3).

Rebuilt from the published rules; the thermal-clock core follows the authors' clock audit and is checked against the case
tables of the study.

Thermal clock. Growing degree days accumulate from the sowing day: GDD = sum over UTC days of max(T_mean - T_base, 0), where
T_mean is the mean of the 24 hourly 2 m air temperatures of the day; a day with a missing hour is invalid. Threshold k is
reached on the first day d whose cumulative GDD is >= the threshold, provided every day from sowing through d is valid; a
threshold that is not reached before harvest, or whose crossing follows an invalid day, is unknown.

Windows and decisions. The window of decision time k runs from sowing to 00:00 UTC of the day after the threshold day; the
harvest-time window runs to 00:00 UTC of the day after harvest. A decision is made DECISION_DELAY_DAYS after its window
ends (the assumed availability delay of ERA5-Land, the latest of the inputs). Yield labels are available at 00:00 UTC of
the harvest date plus LABEL_DELAY_DAYS.

Exclusions: an in-season case whose threshold is unknown or whose decision falls on or after the harvest date; every case of
a field-season whose sowing date has no usable clock (for example a date derived from the harvest date) or that has no
eligible in-season case.
"""
import numpy as np
import pandas as pd

DAY_MS = 86_400_000
DECISION_DELAY_DAYS = 6
LABEL_DELAY_DAYS = 30
# crop: (base temperature in degrees C, GDD thresholds of the four in-season decision times)
CROPS = {
    'corn': (10.0, (400.0, 800.0, 1200.0, 1600.0)),
    'soybean': (10.0, (400.0, 800.0, 1200.0, 1600.0)),
    'wheat': (0.0, (600.0, 1200.0, 1800.0, 2400.0)),
    'rapeseed': (0.0, (700.0, 1400.0, 2100.0, 2800.0)),
}
STAGES = ('tau1', 'tau2', 'tau3', 'tau4')
CLOCK_UNAVAILABLE_SOWING = ('290 days before harvest',)   # sowing dates derived from the harvest date


def daily_mean(hourly):
    """[..., days, 24] hourly values of consecutive UTC days -> [..., days] daily mean; NaN when any hour is missing."""
    h = np.asarray(hourly, dtype=np.float64)
    return np.where(np.isfinite(h).all(-1), h.mean(-1), np.nan)


def threshold_days(tmean_c, sowing_day, harvest_day, first_day, t_base, thresholds):
    """Day number on which each threshold is first reached ([seasons, thresholds], NaN = unknown).

    tmean_c [seasons, n]: daily mean temperature (degrees C) of the UTC days first_day[s] + 0..n-1 (NaN = invalid day).
    Days before sowing or after harvest are ignored. A crossing counts only when every day from sowing through the crossing
    day is valid.
    """
    t = np.asarray(tmean_c, dtype=np.float64)
    days = np.asarray(first_day)[:, None] + np.arange(t.shape[1])[None, :]
    inside = (days >= np.asarray(sowing_day)[:, None]) & (days <= np.asarray(harvest_day)[:, None])
    valid = inside & np.isfinite(t)
    gdd = np.where(valid, np.maximum(t - t_base, 0.0), 0.0)
    cum = np.cumsum(gdd, axis=1)
    hit = valid[:, :, None] & (cum[:, :, None] >= np.asarray(thresholds)[None, None, :])
    exists = hit.any(1)
    ix = hit.argmax(1)
    day = np.take_along_axis(days, ix, axis=1)
    n_valid = np.take_along_axis(np.cumsum(valid, axis=1), ix, axis=1)
    complete = n_valid == day - np.asarray(sowing_day)[:, None] + 1
    return np.where(exists & complete, day, np.nan)


def schedule(seasons, tmean_c, first_day):
    """Case table of field-seasons: one row per season and decision time (tau1..tau4, harvest).

    seasons: columns season_id, crop, sowing_day, harvest_day (UTC day numbers since 1970-01-01) and sowing_type.
    tmean_c, first_day: as in threshold_days(), one row per season in the order of seasons.
    Returns columns season_id, stage, threshold_day, window_end_ms, decision_ms, label_ms, eligible, reason.
    """
    S = seasons.reset_index(drop=True)
    rows = []
    for crop, g in S.groupby('crop', sort=False):
        base, thr = CROPS[crop]
        ix = g.index.to_numpy()
        td = threshold_days(np.asarray(tmean_c)[ix], g.sowing_day.to_numpy(), g.harvest_day.to_numpy(),
                            np.asarray(first_day)[ix], base, thr)
        for k, stage in enumerate(STAGES):
            rows.append(pd.DataFrame(dict(row=ix, stage=stage, threshold_day=td[:, k])))
    T = pd.concat(rows, ignore_index=True)
    T = pd.concat([T, pd.DataFrame(dict(row=S.index.to_numpy(), stage='harvest',
                                        threshold_day=np.nan))], ignore_index=True)
    s = S.loc[T.row]
    harvest = s.harvest_day.to_numpy()
    end_day = np.where(T.stage == 'harvest', harvest + 1, T.threshold_day + 1)
    T['window_end_ms'] = end_day * DAY_MS
    T['decision_ms'] = T.window_end_ms + DECISION_DELAY_DAYS * DAY_MS
    T['label_ms'] = (harvest + LABEL_DELAY_DAYS) * DAY_MS
    inseason = (T.stage != 'harvest').to_numpy()
    unknown = inseason & ~np.isfinite(T.threshold_day.to_numpy())
    late = inseason & ~unknown & (T.decision_ms.to_numpy() >= harvest * DAY_MS)
    no_clock = s.sowing_type.isin(CLOCK_UNAVAILABLE_SOWING).to_numpy()
    reason = np.select([no_clock, unknown, late], ['sowing_without_clock', 'threshold_unknown', 'decision_on_or_after_harvest'],
                       '')
    # a field-season without any eligible in-season case is dropped, harvest-time case included
    ok_in = pd.Series(inseason & (reason == '')).groupby(T.row.to_numpy()).any()
    reason = np.where(~inseason & (reason == '') & ~ok_in.loc[T.row].to_numpy(), 'no_eligible_decision', reason)
    T['eligible'] = reason == ''
    T['reason'] = reason
    T.insert(0, 'season_id', s.season_id.to_numpy())
    return T.drop(columns='row')
