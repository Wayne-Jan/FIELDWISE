"""Season-date proxies and candidate areas of one season, from Sentinel-2 L2A B4, B8, SCL.

NDVI processing (SCL 4, 5 clear sky; daily
interpolation over gaps <= 45 days; 15-day moving average; peak, SOS20 and EOS20 within 180 days of the peak; amplitude >=
0.1), with no date offsets and no duration thresholds:
  noncrop       highest smoothed NDVI < 0.4
  no_cycle      no peak in the autumn-winter window Y-10-15..(Y+1)-04-15 with SOS20 and EOS20
  low_cover     peak NDVI < 0.7
  flooded       share of SCL water within 30 days of SOS20 >= 0.2
  other_season  SOS20 outside Y-08-01..Y-12-31 (the cycle does not start in the autumn of year Y)
  candidate     otherwise: an autumn-winter growth cycle with high cover, not flooded. This does not identify maize; the
                tutorial calls these "image-screened autumn-winter growth-cycle candidate areas, run with maize parameters;
                crop identity not verified". OSM parcels are coarse blocks, so the connected areas are not confirmed
                management fields either.
SOS20 and EOS20 are kept as season start and end proxies (not sowing or harvest dates), with the clear observations and the
longest gap between clear observations from SOS20 to EOS20.
Fields: candidate pixels split by OSM farmland parcels (osm_id), 4-connected, at least 20 pixels (pixels outside every parcel
form their own fields). Unit selection: fields ordered by distance of their centre to (cx, cy), then field id; taken in that
order up to n fields while the pixel total stays within max_pixels.

usage: python -m fieldwise.collect.season_proxies <raw_s2_dir> <parcels_gpkg> <season_year Y> <cx> <cy> <n> <max_pixels> <out_dir>
(example tool: how the Tainan demonstration obtained its fields and season dates; with your own fields and dates skip it)
"""
import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy import ndimage

RAW, GPKG, Y, CX, CY, N, MAXPIX, OUT = (Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]),
                                         float(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), Path(sys.argv[8]))
DAY_MS = 86_400_000
MAX_GAP, SMOOTH, SEARCH, FRAC, MIN_AMP = 45, 15, 180, 0.2, 0.1
MIN_PEAK, WATER, NONCROP, MIN_PIX = 0.7, 0.2, 0.4, 20
CLASSES = ['noncrop', 'no_cycle', 'low_cover', 'flooded', 'other_season', 'candidate']
DATE_SOURCE = 'S2 L2A NDVI SOS20/EOS20 (retrospective, whole-season imagery)'


def days(s):
    return int((np.datetime64(s) - np.datetime64('1970-01-01')).astype(int))


def gaps(M):
    """Previous and next observed day of every day (-1 / nd when none)."""
    nd = M.shape[1]
    ar = np.arange(nd)
    fin = np.isfinite(M)
    prv = np.maximum.accumulate(np.where(fin, ar, -1), axis=1)
    nxt = np.minimum.accumulate(np.where(fin, ar, nd)[:, ::-1], axis=1)[:, ::-1]
    return prv, nxt


def smooth(M):
    nd = M.shape[1]
    ar = np.arange(nd)
    prv, nxt = gaps(M)
    ok = (prv >= 0) & (nxt < nd) & (nxt - prv <= MAX_GAP)
    pv = np.take_along_axis(M, np.clip(prv, 0, nd - 1), 1)
    nv = np.take_along_axis(M, np.clip(nxt, 0, nd - 1), 1)
    w = np.where(nxt > prv, (ar - prv) / np.maximum(nxt - prv, 1), 0.0)
    Yv = np.where(ok, pv + w * (nv - pv), np.nan).astype(np.float32)
    h = SMOOTH // 2
    fin = np.isfinite(Yv)
    cs = np.concatenate([np.zeros((len(M), 1), np.float32), np.cumsum(np.where(fin, Yv, 0), 1, dtype=np.float32)], 1)
    cn = np.concatenate([np.zeros((len(M), 1), np.float32), np.cumsum(fin, 1, dtype=np.float32)], 1)
    lo, hi = np.clip(ar - h, 0, nd), np.clip(ar + h + 1, 0, nd)
    return np.where(fin, (cs[:, hi] - cs[:, lo]) / np.maximum(cn[:, hi] - cn[:, lo], 1), np.nan)


def save(obj, name):
    tmp = OUT / f'.{name}.tmp'
    if name.endswith('.parquet'):
        obj.to_parquet(tmp, index=False)
    elif name.endswith('.npy'):
        with open(tmp, 'wb') as f:
            np.save(f, obj)
    else:
        tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, OUT / name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    G = json.loads((RAW / 'GRID.json').read_text())
    meta = pd.DataFrame(json.loads((RAW / 'IMAGES.json').read_text())).sort_values('time_ms').reset_index(drop=True)
    meta['day'] = meta.time_ms // DAY_MS
    st = [np.load(RAW / f"{i.split('/')[-1]}.npz") for i in meta.id]
    B4 = np.stack([s['B4'] for s in st]).reshape(len(st), -1).astype(np.float32)
    B8 = np.stack([s['B8'] for s in st]).reshape(len(st), -1).astype(np.float32)
    SC = np.stack([s['SCL'] for s in st]).reshape(len(st), -1)
    clear = np.isin(SC, (4, 5)) & (B4 > 0) & (B8 > 0)
    ndvi = np.where(clear, (B8 - B4) / np.maximum(B8 + B4, 1), np.nan).astype(np.float32)
    water = np.where(np.isin(SC, (4, 5, 6)), SC == 6, np.nan).astype(np.float32)
    d0 = int(meta.day.min()) - SEARCH
    nd = int(meta.day.max()) - d0 + SEARCH + 1
    col = (meta.day.to_numpy() - d0).astype(int)
    uniq, inv = np.unique(col, return_inverse=True)                         # same day, several tiles: mean
    ws, we = days(f'{Y}-10-15') - d0, days(f'{Y + 1}-04-15') - d0
    s0, s1 = days(f'{Y}-08-01') - d0, days(f'{Y}-12-31') - d0
    P = ndvi.shape[1]
    x = ndvi.T
    num = np.zeros((P, len(uniq)), np.float32)
    den = np.zeros((P, len(uniq)), np.float32)
    fx = np.isfinite(x)
    np.add.at(num.T, inv, np.where(fx, x, 0).T)
    np.add.at(den.T, inv, fx.T.astype(np.float32))
    M = np.full((P, nd), np.nan, np.float32)
    M[:, uniq] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    S = smooth(M)
    dd = np.arange(nd)[None, :]
    Sw = np.where((dd >= ws) & (dd <= we) & np.isfinite(S), S, -np.inf)
    pk = Sw.argmax(1)
    pval = Sw[np.arange(P), pk]
    has = np.isfinite(pval)
    left = (dd >= (pk - SEARCH)[:, None]) & (dd <= pk[:, None])
    right = (dd >= pk[:, None]) & (dd <= (pk + SEARCH)[:, None])
    bl = np.min(np.where(left & np.isfinite(S), S, np.inf), 1)
    br = np.min(np.where(right & np.isfinite(S), S, np.inf), 1)
    al, ar_ = pval - bl, pval - br
    bel = left & (dd < pk[:, None]) & (S <= (bl + FRAC * al)[:, None])
    ber = right & (dd > pk[:, None]) & (S <= (br + FRAC * ar_)[:, None])
    sos = np.where(bel.any(1), nd - 1 - bel[:, ::-1].argmax(1) + 1, -1)
    eos = np.where(ber.any(1), ber.argmax(1), -1)
    okc = has & (al >= MIN_AMP) & (ar_ >= MIN_AMP) & (sos >= 0) & (eos >= 0)
    near = (np.abs(col[None, :] - sos[:, None]) <= 30) & (sos[:, None] >= 0)
    wt = water.T
    wf = np.where(near & np.isfinite(wt), wt, 0).sum(1) / np.maximum((near & np.isfinite(wt)).sum(1), 1)
    peak_any = np.nanmax(np.where(np.isfinite(S), S, -np.inf), 1)
    cls = np.select([peak_any < NONCROP, ~okc, pval < MIN_PEAK, wf >= WATER, (sos < s0) | (sos > s1)], range(5), 5)
    # observation quality from SOS20 to EOS20: clear observation days and the longest gap between them
    span = (dd >= sos[:, None]) & (dd <= eos[:, None]) & okc[:, None]
    n_clear = (np.isfinite(M) & span).sum(1)
    prv, nxt = gaps(M)
    gap = np.where(span & (prv >= 0) & (nxt < nd), nxt - prv, 0).max(1)
    h, w = G['h'], G['w']
    grids = dict(cls=cls, sos=np.where(okc, sos + d0, -1), eos=np.where(okc, eos + d0, -1), n_clear=n_clear, max_gap=gap)
    for k, v in grids.items():
        save(v.reshape(h, w).astype(np.int32), f'{k}.npy')

    # candidate fields split by OSM parcels
    cand = (cls == CLASSES.index('candidate')).reshape(h, w)
    par = gpd.read_file(GPKG).to_crs(G['crs']).reset_index(drop=True)
    cy_, cx_ = np.mgrid[0:h, 0:w]
    px, py = G['x0'] + (cx_.ravel() + 0.5) * 10, G['y1'] - (cy_.ravel() + 0.5) * 10
    pid = np.zeros(h * w, np.int32)
    for i, geom in enumerate(par.geometry):                                 # a few dozen parcels
        pid[shapely.contains_xy(geom, px, py)] = i + 1
    pid = pid.reshape(h, w)
    fid = np.full((h, w), -1, np.int64)
    nxt_id = 0
    four = ndimage.generate_binary_structure(2, 1)
    for p in np.unique(pid[cand]):                                          # parcels with candidates (0 = outside)
        lab, n = ndimage.label(cand & (pid == p), structure=four)
        size = np.bincount(lab.ravel())
        keep = np.flatnonzero(size >= MIN_PIX)
        keep = keep[keep > 0]
        lut = np.full(n + 1, -1, np.int64)
        lut[keep] = nxt_id + np.arange(len(keep))
        sel = lab > 0
        fid[sel] = np.where(lut[lab[sel]] >= 0, lut[lab[sel]], fid[sel])
        nxt_id += len(keep)
    rr, cc = np.nonzero(fid >= 0)
    xs, ys = G['x0'] + (cc + 0.5) * 10, G['y1'] - (rr + 0.5) * 10
    lon, lat = Transformer.from_crs(G['crs'], 'EPSG:4326', always_xy=True).transform(xs, ys)
    flat = rr * w + cc
    pix = pd.DataFrame(dict(field=fid[rr, cc], row=rr, col=cc, x=xs, y=ys, lon=lon, lat=lat, sos=grids['sos'][flat],
                            eos=grids['eos'][flat], n_clear=n_clear[flat], max_gap=gap[flat], parcel=pid[rr, cc]))
    g = pix.groupby('field')
    osm = np.r_[0, par.osm_id.to_numpy()]
    F = pd.DataFrame(dict(n_pix=g.size(), x=g.x.mean(), y=g.y.mean(), lon=g.lon.mean(), lat=g.lat.mean(),
                          season_start_proxy_day=g.sos.median().round().astype(int),
                          season_end_proxy_day=g.eos.median().round().astype(int),
                          sos_iqr_days=g.sos.quantile(.75) - g.sos.quantile(.25),
                          eos_iqr_days=g.eos.quantile(.75) - g.eos.quantile(.25),
                          n_clear_median=g.n_clear.median(), max_gap_days=g.max_gap.max(),
                          osm_id=osm[g.parcel.first().to_numpy()])).reset_index()
    F['date_source'] = DATE_SOURCE
    F['dist_m'] = np.hypot(F.x - CX, F.y - CY)
    order = F.sort_values(['dist_m', 'field']).reset_index(drop=True)
    take, tot = [], 0
    for r in order.itertuples():                                            # at most n fields
        if len(take) == N:
            break
        if tot + r.n_pix <= MAXPIX:
            take.append(r.field)
            tot += r.n_pix
    U = order[order.field.isin(take)].reset_index(drop=True)
    save(F, 'ALL_FIELDS.parquet')
    save(U, 'FIELDS.parquet')
    save(pix[pix.field.isin(take)].reset_index(drop=True), 'PIXELS.parquet')
    summ = dict(season_year=Y, classes={k: int((cls == i).sum()) for i, k in enumerate(CLASSES)}, candidate_fields=len(F),
                unit_fields=len(U), unit_pixels=int(U.n_pix.sum()), selection=dict(centre=[CX, CY], n=N, max_pixels=MAXPIX),
                date_source=DATE_SOURCE, images=len(meta), grid=G, min_pix=MIN_PIX,
                unit_season_start=[str(pd.to_datetime(U.season_start_proxy_day.min(), unit='D').date()),
                                   str(pd.to_datetime(U.season_start_proxy_day.max(), unit='D').date())],
                unit_season_end=[str(pd.to_datetime(U.season_end_proxy_day.min(), unit='D').date()),
                                 str(pd.to_datetime(U.season_end_proxy_day.max(), unit='D').date())])
    save(summ, 'FIELDS.json')
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
