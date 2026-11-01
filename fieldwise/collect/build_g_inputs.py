"""Public maize data workflow: build_g_inputs. Run through python -m fieldwise.collect; see docs/DATA_COLLECTION.md."""
import json
import os
import sys
from pathlib import Path

from .settings import config, project, weather_cells, month_edges
import numpy as np
import pandas as pd

BI, RAW, FD, LD, OUT = map(Path, sys.argv[1:6])
OUT.mkdir(parents=True, exist_ok=True)
DAY, HOUR, PRE = 86_400_000, 3_600_000, 30
BANDS = ['temperature_2m', 'dewpoint_temperature_2m', 'total_precipitation_hourly', 'surface_solar_radiation_downwards_hourly',
         'surface_thermal_radiation_downwards_hourly', 'surface_net_solar_radiation_hourly', 'surface_net_thermal_radiation_hourly',
         'u_component_of_wind_10m', 'v_component_of_wind_10m', 'surface_pressure', 'potential_evaporation_hourly',
         'total_evaporation_hourly', 'evaporation_from_vegetation_transpiration_hourly', 'volumetric_soil_water_layer_1',
         'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4', 'skin_temperature',
         'soil_temperature_level_1']
FLUX = np.array([b.endswith('_hourly') for b in BANDS])
THEAT = 30.0


def save(path, fn, obj):
    tmp = path.with_name('.tmp.' + path.name)
    fn(tmp, obj)
    os.replace(tmp, path)


def es(t):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


C = pd.read_parquet(BI / 'CASES.parquet')
F = pd.read_parquet(FD / 'FIELDS.parquet').assign(fid=lambda d: 900_000 + d.field).set_index('fid')
wx = pd.read_parquet(RAW / 'era5/ERA5_HOURLY.parquet')
hours = np.array(sorted(wx.time.unique()), np.int64)
cellv = {}
for (lo, la), g in wx.groupby(['cell_lon', 'cell_lat']):                    # nine cells
    v = g.set_index('time').reindex(hours)[BANDS].to_numpy(float, copy=True)
    v[v <= -1e14] = np.nan
    j = BANDS.index('total_precipitation_hourly')
    p = v[:, j]
    p[(p < 0) & (p >= -1e-6)] = 0.0
    p[p < -1e-6] = np.nan
    cellv[(round(lo, 1), round(la, 1))] = v


def corners(lon, lat):
    lo0, la0 = np.floor(lon * 10) / 10, np.floor(lat * 10) / 10
    tx, ty = (lon - lo0) * 10, (lat - la0) * 10
    return [((round(lo0, 1), round(la0, 1)), (1 - tx) * (1 - ty)), ((round(lo0 + .1, 1), round(la0, 1)), tx * (1 - ty)),
            ((round(lo0, 1), round(la0 + .1, 1)), (1 - tx) * ty), ((round(lo0 + .1, 1), round(la0 + .1, 1)), tx * ty)]


def renorm(cw):
    """[hours, 19] weight-renormalised mean over cells with a valid value (g_pack / e_ridge)"""
    num, den = np.zeros((len(hours), len(BANDS))), np.zeros((len(hours), len(BANDS)))
    for k, w in cw:
        if w > 0:
            x = cellv[k]
            ok = np.isfinite(x)
            num += np.where(ok, x, 0) * w
            den += ok * w
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def strict_t(cw):
    """field hourly 2 m temperature, NaN unless every positive-weight cell is valid (daily_t2m_extremes / build_inputs)"""
    acc, good = np.zeros(len(hours)), np.ones(len(hours), bool)
    for k, w in cw:
        if w > 0:
            x = cellv[k][:, 0]
            acc += w * np.nan_to_num(x)
            good &= np.isfinite(x)
    return np.where(good, acc, np.nan)


# LST: per-pixel MODIS cell of every service pixel, raw daily values per cell
pc = pd.read_parquet(LD / 'PIXEL_CELL.parquet')
pc['fid'] = 900_000 + pc.field
lst = pd.read_parquet(LD / 'LST_DAILY.parquet')


def lst_field(fid, part):
    cells = pc[pc.fid == fid].groupby(['cell_row', 'cell_col']).size().rename('npx')
    sub = lst.merge(cells.reset_index(), on=['cell_row', 'cell_col'])
    raw = sub[f'LST_{part}_1km'].to_numpy()
    qc = sub[f'QC_{part}'].to_numpy()
    ok = (raw != 0) & ((qc & 3) <= 1) & (((qc >> 6) & 3) <= 1)
    t = raw * 0.02 - 273.15
    sub = sub.assign(okw=ok * sub.npx, val=np.where(ok, t, 0) * sub.npx)
    g = sub.groupby('day')[['okw', 'val']].sum()
    cov = g.okw / cells.sum()
    return pd.Series(np.where(cov >= 0.5, g.val / np.maximum(g.okw, 1e-9), np.nan), index=g.index)


X_list, fs_rows, case_rows, e_rows, l_rows = [], [], [], [], []
for i, (fid, cs) in enumerate(C.groupby('fid')):                              # 292 field-seasons
    f = F.loc[fid]
    cw = corners(f.lon, f.lat)
    v = renorm(cw)
    seed = int(f.sow_day)
    start = seed - PRE
    E = cs.window_end_utc_ms.to_numpy(np.int64)
    Cd = cs.decision_utc_ms.to_numpy(np.int64)
    assert (Cd % DAY == 0).all() and (E == Cd - 6 * DAY).all()
    n = (E // DAY - start).astype(int)
    L = int(n.max())
    # pack: slot of an instantaneous band = hour of t; of an accumulation = hour of t - 1 h; days [start, start + L)
    X = np.full((L, 24, len(BANDS)), np.nan, np.float32)
    for fl in (False, True):
        cols = np.flatnonzero(FLUX == fl)
        st = hours - (HOUR if fl else 0)
        d = st // DAY - start
        h = (st % DAY) // HOUR
        m = (d >= 0) & (d < L) & (hours < E.max() + (HOUR if fl else 0))      # instantaneous t < E; accumulation t <= E
        X[d[m][:, None], h[m][:, None], cols[None, :]] = v[m][:, cols]
    X_list.append(X)
    fs_rows.append(dict(fid=int(fid), year=config()['year'], seed_day=seed, start_day=start, n_days_max=L))
    case_rows += [dict(case_id=c, fs_row=i, n_days=int(k)) for c, k in zip(cs.case_id, n)]
    # E_ridge daily table from seeding
    T, Td = v[:, 0] - 273.15, v[:, 1] - 273.15
    df = pd.DataFrame(dict(day=hours // DAY, heat=np.maximum(T - THEAT, 0) / 24, vpd2=(es(T) - es(Td) > 2).astype(float),
                           rz=.07 * v[:, 13] + .21 * v[:, 14] + .72 * v[:, 15]))
    df.loc[~np.isfinite(T), 'heat'] = np.nan
    df.loc[~(np.isfinite(T) & np.isfinite(Td)), 'vpd2'] = np.nan
    gd = df.groupby('day').agg(heat_dd=('heat', 'sum'), vpd_hours=('vpd2', 'sum'), theta_rz=('rz', 'mean'))
    gd['precip_mm'] = pd.Series(v[:, 2] * 1000, index=(hours - HOUR) // DAY).groupby(level=0).sum().reindex(gd.index)
    gd = gd[gd.index >= seed]
    days = gd.index.to_numpy()
    cum = lambda x: np.concatenate([[0.], np.cumsum(np.nan_to_num(x))])         # noqa: E731
    usable = E // DAY - 1
    hi = np.searchsorted(days, usable, side='right')
    lo14 = np.searchsorted(days, usable - 14, side='right')
    lo7 = np.maximum(hi - 7, 0)
    rz = gd.theta_rz.to_numpy()
    fr = np.isfinite(rz)
    Sr, Nr = cum(np.where(fr, rz, 0)), cum(fr)
    rmin = np.concatenate([[np.nan], np.fmin.accumulate(rz)])
    H, V, P = cum(gd.heat_dd.to_numpy()), cum(gd.vpd_hours.to_numpy()), cum(gd.precip_mm.to_numpy())
    n7 = Nr[hi] - Nr[lo7]
    with np.errstate(invalid='ignore', divide='ignore'):
        last7 = np.where(n7 > 0, (Sr[hi] - Sr[lo7]) / n7, np.nan)
    e_rows.append(pd.DataFrame(dict(case_id=cs.case_id.to_numpy(), E1_total=H[hi], E1_14d=H[hi] - H[lo14], E2_total=V[hi],
                                    E2_14d=V[hi] - V[lo14], E4_cum_precip=P[hi], E3_theta_last7=last7, E3_theta_min=rmin[hi], wx_days=hi)))
    # LST indicators
    Ts = strict_t(cw)
    kd = hours // DAY
    full = pd.Series(np.isfinite(Ts)).groupby(kd).sum() == 24
    tmax = pd.Series(Ts).groupby(kd).max()[full] - 273.15
    ldy, lnt = lst_field(fid, 'Day'), lst_field(fid, 'Night')
    for c in cs.itertuples():
        e_day = int(c.window_end_utc_ms) // DAY
        dd = np.arange(seed, e_day)                                              # days seeding .. E - 1
        ld, ln = ldy.reindex(dd).to_numpy(), lnt.reindex(dd).to_numpy()
        dT = ld - tmax.reindex(dd).to_numpy()
        l14, l30 = dd >= e_day - 14, dd >= e_day - 30
        m = lambda x, s: float(np.nanmean(x[s])) if np.isfinite(x[s]).any() else np.nan   # noqa: E731
        v30 = np.isfinite(dT) & l30
        l_rows.append(dict(case_id=c.case_id, lst_day_14=m(ld, l14), dT_day_14=m(dT, l14), dT_day_seas=m(dT, np.ones_like(l14)),
                           dT_pos_30=float((dT[v30] > 0).mean()) if v30.any() else np.nan, lst_night_14=m(ln, l14),
                           n_day_valid=int(np.isfinite(ld).sum()), n_obs=len(dd)))

Lmax = max(x.shape[0] for x in X_list)
XA = np.full((len(X_list), Lmax, 24, len(BANDS)), np.nan, np.float32)
for i, x in enumerate(X_list):
    XA[i, :len(x)] = x
save(OUT / 'X.npy', np.save, XA)
save(OUT / 'fs.parquet', lambda p, o: o.to_parquet(p, index=False), pd.DataFrame(fs_rows))
save(OUT / 'cases.parquet', lambda p, o: o.to_parquet(p, index=False), pd.DataFrame(case_rows))
save(OUT / 'PACK.json', lambda p, o: p.write_text(json.dumps(o, indent=1)), dict(bands=BANDS, field_seasons=len(fs_rows), L_max=Lmax,
                                                                                  pre_seeding_days=PRE))
Etab = pd.concat(e_rows).set_index('case_id')
Ltab = pd.DataFrame(l_rows).set_index('case_id')
save(OUT / 'E_TAIWAN.parquet', lambda p, o: o.to_parquet(p), Etab)
save(OUT / 'LST_TAIWAN.parquet', lambda p, o: o.to_parquet(p), Ltab)
print(json.dumps(dict(field_seasons=len(fs_rows), cases=len(case_rows), L_max=Lmax,
                      E_median=Etab.median().round(2).to_dict(), LST_missing=Ltab.drop(columns=['n_day_valid', 'n_obs']).isna().mean().round(3).to_dict(),
                      LST_median=Ltab.median().round(2).to_dict()), indent=1))

