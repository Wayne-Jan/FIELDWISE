"""Public maize data workflow: build_base_inputs. Run through python -m fieldwise.collect; see docs/DATA_COLLECTION.md."""
import json
import os
import sys
from pathlib import Path

from .settings import config, project, weather_cells, month_edges
import numpy as np
import pandas as pd

FD, RAW, TER, OUT = map(Path, sys.argv[1:5])
(OUT / 's2_pack').mkdir(parents=True, exist_ok=True)
DAY, HOUR = 86_400_000, 3_600_000
WX_LAT, S2_LAT = 6 * DAY, 24 * HOUR
TBASE, THR = 10.0, {'tau1': 400, 'tau2': 800, 'tau3': 1200, 'tau4': 1600}
FID0, T = 900_000, config()['year']
NB, SCL_PAD = 12, 12
SOIL_PROPS = ('cec', 'cfvo', 'clay', 'nitrogen', 'phh2o', 'sand', 'silt', 'soc')
SOIL_DEPTHS = ('0-5', '5-15', '15-30')
S2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']


def save(path, fn, obj):
    tmp = path.with_name('.tmp.' + path.name)                    # keep the extension (np.save / np.savez add one otherwise)
    fn(tmp, obj)
    os.replace(tmp, path)


F = pd.read_parquet(FD / 'FIELDS.parquet').sort_values('field').reset_index(drop=True)
PX = pd.read_parquet(FD / 'PIXELS.parquet').sort_values(['field', 'row', 'col']).reset_index(drop=True)
F['fid'] = FID0 + F.field
n_pix = PX.groupby('field').size().reindex(F.field).to_numpy()
pix0 = np.r_[0, np.cumsum(n_pix)]

# ---- weather: hourly field series (bilinear over the 4 surrounding cells) ----
wx = pd.read_parquet(RAW / 'era5/ERA5_HOURLY.parquet')
cells = {(lo, la): g.set_index('time').sort_index() for (lo, la), g in wx.groupby(['cell_lon', 'cell_lat'])}
hours = np.array(sorted(wx.time.unique()), np.int64)
t2 = {k: g.temperature_2m.reindex(hours).to_numpy(float, copy=True) for k, g in cells.items()}
pr = {k: g.total_precipitation_hourly.reindex(hours).to_numpy(float, copy=True) for k, g in cells.items()}
for x in list(t2.values()) + list(pr.values()):                 # loader era5_land cleaning: fill -> NaN
    x[x <= -1e14] = np.nan
for p in pr.values():                                          # negative precipitation: >= -1e-6 -> 0, below -> invalid
    p[(p < 0) & (p >= -1e-6)] = 0.0
    p[p < -1e-6] = np.nan


def bilinear(lon, lat):
    lo0, la0 = np.floor(lon * 10) / 10, np.floor(lat * 10) / 10
    tx, ty = (lon - lo0) * 10, (lat - la0) * 10
    return [((round(lo0, 1), round(la0, 1)), (1 - tx) * (1 - ty)), ((round(lo0 + .1, 1), round(la0, 1)), tx * (1 - ty)),
            ((round(lo0, 1), round(la0 + .1, 1)), (1 - tx) * ty), ((round(lo0 + .1, 1), round(la0 + .1, 1)), tx * ty)]


def field_hourly(series, wts):
    acc, good = np.zeros(len(hours)), np.ones(len(hours), bool)
    for k, w in wts:
        if w > 0:
            x = series[k]
            acc += w * np.nan_to_num(x)
            good &= np.isfinite(x)
    return np.where(good, acc, np.nan)


d_all = np.arange(hours.min() // DAY, hours.max() // DAY + 1)
d0 = d_all[0]


def daily(T2, P):
    out = np.full((len(d_all), 4), np.nan)
    kt = hours // DAY - d0
    kp = (hours - HOUR) // DAY - d0
    for k, x, kind in ((kt, T2, 'T'), (kp, P, 'P')):
        m = (k >= 0) & (k < len(d_all)) & np.isfinite(x)
        full = np.bincount(k[m], minlength=len(d_all)) == 24
        if kind == 'T':
            s = np.bincount(k[m], weights=x[m], minlength=len(d_all))
            mx, mn = np.full(len(d_all), -np.inf), np.full(len(d_all), np.inf)
            np.maximum.at(mx, k[m], x[m])
            np.minimum.at(mn, k[m], x[m])
            out[full, 0], out[full, 1], out[full, 2] = s[full] / 24, mx[full], mn[full]
        else:
            out[full, 3] = np.bincount(k[m], weights=x[m], minlength=len(d_all))[full]
    return out.astype(np.float32)


# ---- S2 images ----
imgs = pd.DataFrame(json.loads((RAW / 's2/IMAGES.json').read_text())).sort_values(['time_ms', 'tile']).reset_index(drop=True)
t_img = imgs.time_ms.to_numpy(np.int64)
assert (np.diff(t_img) >= 0).all()                     # same acquisition from two tiles: two images (as the unified dataset)
rr, cc = PX.row.to_numpy(), PX.col.to_numpy()
stack = [np.load(RAW / 's2' / f'{i}.npz') for i in imgs.image_id]               # about 80 small files, each read once
V = np.stack([np.stack([s[b][rr, cc] for b in S2_BANDS], -1) for s in stack], 1)  # [pix, img, 12]
S = np.stack([s['SCL'][rr, cc] for s in stack], 1)                                 # [pix, img]
OK = (V != 0).all(-1)
SCL = np.where(S != 0, S, SCL_PAD).astype(np.uint8)

# ---- cases per field ----
rows, wx_pack, img_meta, gdd_log = [], [], [], []
for k, f in F.iterrows():                                                          # 292 fields
    wts = bilinear(f.lon, f.lat)
    T2h, Ph = field_hourly(t2, wts), field_hourly(pr, wts)
    dd = daily(T2h, Ph)
    s_day = int(f.sow_day)
    i_s = s_day - d0
    tm = dd[i_s:, 0] - 273.15
    valid = np.isfinite(tm)
    first_bad = np.flatnonzero(~valid)
    stop = first_bad[0] if len(first_bad) else len(tm)
    cum = np.cumsum(np.maximum(tm[:stop] - TBASE, 0))
    case_rows = []
    for tau, thr in THR.items():
        hit = np.flatnonzero(cum >= thr)
        if not len(hit):
            continue
        d = s_day + int(hit[0])
        E, C = (d + 1) * DAY, (d + 7) * DAY
        if C >= int(f.harvest_day) * DAY:                                         # decision not before the estimated harvest
            continue
        case_rows.append((tau, d, E, C))
    if not case_rows:
        continue
    h = int(f.harvest_day)                                                          # harvest time: window to the end of the season
    case_rows.append(('harvest', h, (h + 1) * DAY, (h + 7) * DAY))
    s_ms = s_day * DAY
    e_max = max(r[2] for r in case_rows)
    days = np.arange(s_day, -(-e_max // DAY))
    wvals = dd[days - d0]
    ds = days * DAY
    for tau, d, E, C in case_rows:
        i0 = int(np.searchsorted(t_img, s_ms, 'left'))
        i1 = int(min(np.searchsorted(t_img, E, 'left'), np.searchsorted(t_img, C - S2_LAT, 'right')))
        w0 = int(np.searchsorted(ds, s_ms, 'left'))
        w1 = int(min(np.searchsorted(ds, E - DAY, 'right'), np.searchsorted(ds, C - DAY - WX_LAT, 'right')))
        rows.append(dict(case_id=f'corn:TW{int(f.field)}:{T}:{tau}', crop='corn', fid=int(f.fid), year=T, stage=tau,
                         first_round='include', thermal_day=d, input_start_utc_ms=s_ms, window_end_utc_ms=E, decision_utc_ms=C,
                         i0=i0, i1=max(i0, i1), w0=w0, w1=max(w0, w1), wx_days_valid=int(np.isfinite(wvals[w0:max(w0, w1)]).all(1).sum()),
                         n_img=len(t_img), n_pix=int(n_pix[k]), pix0=int(pix0[k]), row_start=0))
    wx_pack.append((int(f.fid), days.astype(np.int32), wvals))
    gdd_log.append(dict(field=int(f.field), cases=len(case_rows)))
C_ = pd.DataFrame(rows)
keep_f = F[F.fid.isin(C_.fid)].reset_index(drop=True)

# ---- static, weather and S2 pack for the fields that have cases (fid order) ----
soil_z = np.load(RAW / 'soil_official/SOIL.npz')    # ISRIC official COGs (soil_official.py); the GEE copy's nitrogen is about 10x off
soil_all = np.stack([soil_z[f'{p}_{d}cm_mean'][rr, cc] for p in SOIL_PROPS for d in SOIL_DEPTHS], 1).astype(np.float64)
soil_all = np.where(soil_all == -32768, np.nan, soil_all).astype(np.float32)
tz = np.load(TER / 'TERRAIN.npz')
tpos = pd.Series(np.arange(len(tz['row'])), index=pd.MultiIndex.from_arrays([tz['row'], tz['col']]))
ti = tpos.reindex(pd.MultiIndex.from_arrays([rr, cc])).to_numpy()
assert not np.isnan(ti).any()
ti = ti.astype(int)
dem_all = np.stack([tz['aspect_deg'][ti], tz['laplacian_per_m'][ti], tz['elevation_m'][ti], tz['slope_deg'][ti], tz['twi'][ti]], 1).astype(np.float32)
sel_rows, new_pix0, pack_rows, vals, scls, oks = [], [0], [], [], [], []
off = 0
for k, f in keep_f.iterrows():                                                     # fields with cases; a few hundred small slices
    kk = F.index[F.field == f.field][0]
    a, b = int(pix0[kk]), int(pix0[kk + 1])
    sel_rows.append(np.arange(a, b))
    new_pix0.append(new_pix0[-1] + (b - a))
    cs = C_[C_.fid == f.fid]
    i0f, i1f = int(cs.i0.min()), int(max(cs.i1.max(), cs.i0.min()))
    L = i1f - i0f
    pack_rows.append(dict(crop='corn', fid=int(f.fid), n_pix=b - a, L=L, off=off, i0f=i0f))
    vals.append(V[a:b, i0f:i1f].reshape(-1, NB))
    scls.append(SCL[a:b, i0f:i1f].reshape(-1))
    oks.append(OK[a:b, i0f:i1f].reshape(-1))
    off += (b - a) * L
sel = np.concatenate(sel_rows)
remap = {int(f.fid): new_pix0[k] for k, f in keep_f.iterrows()}
C_['pix0'] = C_.fid.map(remap)
save(OUT / 'corn_static.npz', lambda p, o: np.savez(p, **o), dict(
    fid=keep_f.fid.to_numpy(), pix0=np.array(new_pix0), soil=soil_all[sel], dem=dem_all[sel], target=np.full(len(sel), np.nan, np.float32),
    gt_valid=np.zeros(len(sel), bool), soil_names=np.array([f'{p}_{d}' for p in SOIL_PROPS for d in SOIL_DEPTHS]),
    dem_names=np.array(['aspect', 'curvature', 'dem', 'slope', 'twi'])))
save(OUT / 'corn_grid.npz', lambda p, o: np.savez(p, **o), dict(
    row=rr[sel].astype(np.int32), col=cc[sel].astype(np.int32), fid=keep_f.fid.to_numpy(), pix0=np.array(new_pix0)))
wx_by = {fid: (dy, wv) for fid, dy, wv in wx_pack}
fids = keep_f.fid.tolist()
wd = [wx_by[f] for f in fids]
wstart = np.r_[0, np.cumsum([len(x[0]) for x in wd])]
istart = np.arange(len(fids) + 1) * len(t_img)
save(OUT / 'weather_daily.npz', lambda p, o: np.savez(p, **o), dict(
    fid=np.array(fids), w_start=wstart, day=np.concatenate([x[0] for x in wd]).astype(np.int32), values=np.concatenate([x[1] for x in wd]),
    names=np.array(['Temp_mean', 'Temp_max', 'Temp_min', 'Total_prec']), img_start=istart,
    img_day=np.tile((t_img // DAY).astype(np.int32), len(fids)), img_offset=np.zeros(len(fids) * len(t_img), np.int16)))
save(OUT / 's2_pack/corn_vals.npy', np.save, np.concatenate(vals).astype(np.uint16))
save(OUT / 's2_pack/corn_scl.npy', np.save, np.concatenate(scls).astype(np.uint8))
save(OUT / 's2_pack/corn_ok.npy', np.save, np.concatenate(oks).astype(bool))
save(OUT / 's2_pack/PACK_INDEX.parquet', lambda p, o: o.to_parquet(p, index=False), pd.DataFrame(pack_rows))
save(OUT / 'CASES.parquet', lambda p, o: o.to_parquet(p, index=False), C_)
rep = dict(fields=len(keep_f), fields_without_case=int(len(F) - len(keep_f)), cases=len(C_), cases_by_stage=C_.stage.value_counts().to_dict(),
           pixels=int(len(sel)), images=len(t_img), s2_steps_by_stage=C_.assign(n=C_.i1 - C_.i0).groupby('stage').n.median().to_dict(),
           s2_ok_share=float(OK.mean()), wx_days_by_stage=C_.assign(n=C_.w1 - C_.w0).groupby('stage').n.median().to_dict(),
           wx_complete=bool((C_.wx_days_valid == C_.w1 - C_.w0).all()),
           decision_range={s: [str(np.datetime64(int(g.decision_utc_ms.min()), 'ms').astype('datetime64[D]')),
                               str(np.datetime64(int(g.decision_utc_ms.max()), 'ms').astype('datetime64[D]'))] for s, g in C_.groupby('stage')},
           soil_nan_share=float(np.isnan(soil_all[sel]).mean()), dem_nan_share=np.isnan(dem_all[sel]).mean(0).round(4).tolist())
save(OUT / 'BUILD.json', lambda p, o: p.write_text(json.dumps(o, indent=1)), rep)
print(json.dumps(rep, indent=1))

