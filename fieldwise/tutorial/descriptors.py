"""Retrieval descriptors (q features) of the in-season cases, built from the Base inputs of a season.

The formulas are those of the study (field-mean NDVI, NDRE and NDMI of the available season and of each reached
growing-degree-day interval, weather summaries over the same intervals, soil and terrain statistics; docs/INPUTS.md
section 8), copied unchanged. A tau case uses the images and days of its own window; the harvest-time case of the
field-season holds the data. Intervals not reached by the decision are set to missing.
"""
import math

import numpy as np

from ..base.data import DATE_PAD

TAUS = ('tau1', 'tau2', 'tau3', 'tau4')
DAY = 86_400_000
BANDS = dict(B04=3, B05=4, B08=7, B8A=8, B11=10)        # positions in the 12 Sentinel-2 L2A bands of the Base inputs
SCL_VALID_CLASSES = (4, 5, 7)
SOIL = [f'{a}_{d}' for a in ('cec', 'cfvo', 'clay', 'nitrogen', 'phh2o', 'sand', 'silt', 'soc') for d in ('0-5', '5-15', '15-30')]
TERRAIN = ['aspect', 'curvature', 'dem', 'slope', 'twi']
TBASE = {'corn': 10.0, 'soybean': 10.0, 'wheat': 0.0}
EPS = 1e-6


def ratio_index(a, b):
    denom = a + b
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(denom) & (np.abs(denom) > EPS)
    out = np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan, dtype=np.float64)
    np.divide(a - b, denom, out=out, where=valid)
    return np.where(np.isfinite(out), out, np.nan)


def safe_nanmean(values, axis=None):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    count = np.sum(valid, axis=axis)
    total = np.nansum(np.where(valid, values, 0.0), axis=axis)
    out = np.full(np.shape(total), np.nan, dtype=np.float64)
    np.divide(total, count, out=out, where=count > 0)
    return out


def safe_nanstd(values, axis=None):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    count = np.sum(valid, axis=axis)
    mean = safe_nanmean(values, axis=axis)
    if axis is None:
        if int(count) <= 0:
            return np.asarray(np.nan, dtype=np.float64)
        diff2 = np.where(valid, (values - float(mean)) ** 2, 0.0)
        return np.asarray(math.sqrt(float(np.sum(diff2)) / float(count)), dtype=np.float64)
    expanded_mean = np.expand_dims(mean, axis=axis) if isinstance(axis, int) else mean
    diff2 = np.where(valid, (values - expanded_mean) ** 2, 0.0)
    var = np.full(np.shape(mean), np.nan, dtype=np.float64)
    np.divide(np.sum(diff2, axis=axis), count, out=var, where=count > 0)
    return np.sqrt(var)


def finite_or_nan(value):
    try:
        v = float(value)
    except Exception:
        return float('nan')
    return v if math.isfinite(v) else float('nan')


def summarize_vector(prefix, values, out):
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    out[f'{prefix}__mean'] = finite_or_nan(np.mean(finite)) if finite.size else float('nan')
    out[f'{prefix}__std'] = finite_or_nan(np.std(finite)) if finite.size else float('nan')
    out[f'{prefix}__min'] = finite_or_nan(np.min(finite)) if finite.size else float('nan')
    out[f'{prefix}__max'] = finite_or_nan(np.max(finite)) if finite.size else float('nan')


def add_stage_vi_features(row, prefix, vi_ts, stage_ids, valid_time, n_stages):
    for name, series in vi_ts.items():
        valid_series = valid_time & np.isfinite(series)
        row[f'{prefix}__{name}__valid_obs'] = int(np.sum(valid_series))
        row[f'{prefix}__{name}__valid_ratio'] = float(np.mean(valid_series)) if len(valid_series) else float('nan')
        summarize_vector(f'{prefix}__{name}__season', series[valid_series], row)
        row[f'{prefix}__{name}__peak_timestep'] = (int(np.nanargmax(np.where(valid_series, series, np.nan)))
                                                   if np.any(valid_series) else -1)
        for stage in range(n_stages):
            mask = valid_series & (stage_ids == stage)
            values = series[mask]
            sp = f'{prefix}__stage{stage}__{name}'
            summarize_vector(sp, values, row)
            row[f'{sp}__valid_obs'] = int(np.sum(mask))
            row[f'{sp}__valid_ratio'] = float(np.mean(mask)) if len(mask) else float('nan')
            if np.sum(mask) >= 2:
                idx = np.flatnonzero(mask)
                row[f'{sp}__slope'] = finite_or_nan((series[idx[-1]] - series[idx[0]]) / max(1, idx[-1] - idx[0]))
                row[f'{sp}__integral'] = finite_or_nan(np.trapezoid(series[idx], dx=1.0))
            elif np.sum(mask) == 1:
                row[f'{sp}__slope'] = 0.0
                row[f'{sp}__integral'] = finite_or_nan(values[0])
            else:
                row[f'{sp}__slope'] = float('nan')
                row[f'{sp}__integral'] = float('nan')


def max_dry_spell(precip, mask, threshold):
    best = cur = 0
    for is_valid, value in zip(mask, precip):                   # one season of days
        if is_valid and np.isfinite(value) and value <= threshold:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def add_weather_stage_features(row, prefix, weather, weather_dates, stage_ids, valid, n_stages, tbase_c,
                               heat_threshold_c=30.0, cold_threshold_c=5.0, dry_precip_threshold=1e-4):
    tmean, tmax, tmin = (weather[:, j].astype(np.float64) - 273.15 for j in range(3))
    precip = weather[:, 3].astype(np.float64)
    vw = valid & np.isfinite(tmean) & np.isfinite(tmax) & np.isfinite(tmin) & np.isfinite(precip)
    vw &= np.asarray(weather_dates) != DATE_PAD
    gdd = np.where(vw, np.maximum(0.0, tmean - float(tbase_c)), np.nan)
    diurnal = np.where(vw, tmax - tmin, np.nan)
    variables = dict(temp_mean_c=tmean, temp_max_c=tmax, temp_min_c=tmin, diurnal_range_c=diurnal, precip=precip, gdd=gdd)
    for name, values in variables.items():
        summarize_vector(f'{prefix}__{name}__season', values[vw], row)
    row[f'{prefix}__season__gdd_sum'] = finite_or_nan(np.nansum(gdd[vw])) if np.any(vw) else float('nan')
    row[f'{prefix}__season__precip_sum'] = finite_or_nan(np.nansum(precip[vw])) if np.any(vw) else float('nan')
    row[f'{prefix}__season__heat_days'] = int(np.sum(vw & (tmax > heat_threshold_c)))
    row[f'{prefix}__season__cold_days'] = int(np.sum(vw & (tmin < cold_threshold_c)))
    row[f'{prefix}__season__dry_days'] = int(np.sum(vw & (precip <= dry_precip_threshold)))
    row[f'{prefix}__season__dry_spell_max'] = max_dry_spell(precip, vw, dry_precip_threshold)
    for stage in range(n_stages):
        mask = vw & (stage_ids == stage)
        sp = f'{prefix}__stage{stage}'
        row[f'{sp}__valid_days'] = int(np.sum(mask))
        row[f'{sp}__valid_ratio'] = float(np.mean(mask)) if len(mask) else float('nan')
        for name, values in variables.items():
            summarize_vector(f'{sp}__{name}', values[mask], row)
        row[f'{sp}__gdd_sum'] = finite_or_nan(np.nansum(gdd[mask])) if np.any(mask) else float('nan')
        row[f'{sp}__precip_sum'] = finite_or_nan(np.nansum(precip[mask])) if np.any(mask) else float('nan')
        row[f'{sp}__heat_days'] = int(np.sum(mask & (tmax > heat_threshold_c)))
        row[f'{sp}__cold_days'] = int(np.sum(mask & (tmin < cold_threshold_c)))
        row[f'{sp}__dry_days'] = int(np.sum(mask & (precip <= dry_precip_threshold)))
        row[f'{sp}__dry_spell_max'] = max_dry_spell(precip, mask, dry_precip_threshold)


def static_features(soil, dem):
    result = {}
    for j, feature in enumerate(SOIL):
        result[f'q_soil__{feature}__mean'] = finite_or_nan(safe_nanmean(soil[:, j]))
        result[f'q_soil__{feature}__std'] = finite_or_nan(safe_nanstd(soil[:, j]))
    properties = {}
    for j, feature in enumerate(SOIL):
        properties.setdefault(feature.rsplit('_', 1)[0], []).append(j)
    for prop, cols in properties.items():
        result[f'q_soil__{prop}__topsoil_mean'] = finite_or_nan(safe_nanmean(soil[:, cols]))
        result[f'q_soil__{prop}__topsoil_std'] = finite_or_nan(safe_nanstd(soil[:, cols]))
    for j, band in enumerate(TERRAIN):
        result[f'q_topo__{band}__mean'] = finite_or_nan(safe_nanmean(dem[:, j]))
        result[f'q_topo__{band}__std'] = finite_or_nan(safe_nanstd(dem[:, j]))
    return result


def null_future_stage_columns(row, cutoff_index):
    for column in tuple(row):
        for future in range(int(cutoff_index) + 1, len(TAUS)):
            if f'__stage{future}__' in column or column.endswith(f'__stage{future}'):
                row[column] = float('nan')
                break


def field_season(season, cases, crop='corn'):
    """Descriptor rows of the tau cases of one field-season (cases: its rows of the season's case table)."""
    full = cases[cases.stage == 'harvest'].iloc[0]
    taus = cases[cases.stage.isin(TAUS)].set_index('stage')
    x = season.case_inputs(full, full.pix0)
    s2 = x['s2'].astype(np.float64)
    day = x['day']
    valid_s2 = np.isin(x['scl'], np.asarray(SCL_VALID_CLASSES, dtype=np.int16)) & x['ok']
    m = {b: np.where(valid_s2, s2[..., i], np.nan) for b, i in BANDS.items()}
    for v in m.values():
        valid_s2 &= np.isfinite(v) & (v >= 0)
    m = {b: np.where(valid_s2, v, np.nan) for b, v in m.items()}
    vi_pixel = dict(ndvi=ratio_index(m['B08'], m['B04']), ndre=ratio_index(m['B8A'], m['B05']), ndmi=ratio_index(m['B08'], m['B11']))
    vi_ts = {k: safe_nanmean(v, axis=0) for k, v in vi_pixel.items()}
    p0, n = int(full.pix0), int(full.n_pix)
    static = static_features(season.soil[p0:p0 + n].astype(np.float64), season.dem[p0:p0 + n].astype(np.float64))
    wv, wd = x['weather'].astype(np.float64), x['weather_day'].astype(np.int64)
    ends = np.array([float(taus.loc[t, 'window_end_utc_ms']) / DAY if t in taus.index else np.inf for t in TAUS])
    rows = []
    for k, t in enumerate(TAUS):
        if t not in taus.index:
            continue
        c = taus.loc[t]
        s2_valid = np.zeros(len(day), bool)
        s2_valid[int(c.i0) - int(full.i0):int(c.i1) - int(full.i0)] = True
        w_valid = np.zeros(len(wd), bool)
        w_valid[int(c.w0) - int(full.w0):int(c.w1) - int(full.w0)] = True
        s2_stage = np.where(s2_valid, np.clip(np.searchsorted(ends, day, side='right'), 0, 3), -1).astype(np.int16)
        w_stage = np.where(w_valid, np.clip(np.searchsorted(ends, wd, side='right'), 0, 3), -1).astype(np.int16)
        row = dict(case_id=c.case_id)
        add_stage_vi_features(row, 'q_s2_fixedgdd4', vi_ts, s2_stage, s2_valid, 4)
        add_weather_stage_features(row, 'q_weather_fixedgdd4', wv, wd, w_stage, w_valid, 4, TBASE[crop])
        for name, values in vi_pixel.items():
            sup = values[:, s2_valid]
            row[f'q_s2_asof__{name}__within_field_std_mean'] = (finite_or_nan(np.nanmean(safe_nanstd(sup, axis=0)))
                                                               if sup.shape[1] else float('nan'))
        row['q_s2_asof__valid_pixel_time_ratio'] = float(np.mean(valid_s2[:, s2_valid])) if s2_valid.any() else float('nan')
        row.update(static)
        null_future_stage_columns(row, k)
        rows.append(row)
    return rows


def season_descriptors(season, crop='corn'):
    """Descriptor table (one row per tau case) of a season."""
    import pandas as pd                                          # noqa: PLC0415
    rows = []
    for _, g in season.cases.groupby('fid'):                     # field-seasons of the season
        rows += field_season(season, g, crop)
    return pd.DataFrame(rows)
