"""Public maize data workflow: extract_s1_rtc. Run through python -m fieldwise.collect; see docs/DATA_COLLECTION.md."""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from .settings import config, project, weather_cells, month_edges
import numpy as np
import pandas as pd
import rasterio
import requests
from pyproj import Transformer
from rasterio.windows import Window

FD, BI, OUT = map(Path, sys.argv[1:4])
DR = (sys.argv[4], sys.argv[5])
OUT.mkdir(parents=True, exist_ok=True)
STAC = 'https://planetarycomputer.microsoft.com/api/stac/v1/search'
TOKEN_URL = 'https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc'
PAD, NODATA = 40.0, np.float32(-32768)
G = json.loads((FD / 'FIELDS.json').read_text())['grid']
TOK = {}


def request(method, url, **kw):
    """HTTP request with retries on rate limits (429) and server errors (5xx) of the public endpoint."""
    for attempt in range(8):
        r = requests.request(method, url, **kw)
        if r.status_code != 429 and r.status_code < 500:
            break
        time.sleep(min(15 * 2 ** attempt, 600))
    r.raise_for_status()
    return r


def token():
    if not TOK or TOK['expiry'] - time.time() < 600:
        j = request('GET', TOKEN_URL, timeout=60).json()
        TOK.update(token=j['token'], expiry=datetime.fromisoformat(j['msft:expiry'].replace('Z', '+00:00')).timestamp())
    return TOK['token']


def save(name, fn):
    tmp = OUT / f'.tmp.{name}'
    with open(tmp, 'wb') as f:
        fn(f)
    os.replace(tmp, OUT / name)


# ---- service pixels (Base order) -> canonical lon / lat ----
with np.load(BI / 'corn_grid.npz') as z:
    row, col, fids, pix0 = z['row'].astype(np.int64), z['col'].astype(np.int64), z['fid'], z['pix0']
fid_px = np.repeat(fids, np.diff(pix0))
x = G['x0'] + (col + 0.5) * 10.0
y = G['y1'] - (row + 0.5) * 10.0
lon, lat = Transformer.from_crs(G['crs'], 'EPSG:4326', always_xy=True).transform(x, y)
lon, lat = np.asarray(lon, np.float64), np.asarray(lat, np.float64)
save('PIXELS_LONLAT.npz', lambda f: np.savez(f, fid=fid_px, lon=lon, lat=lat, x=x, y=y))

# ---- items ----
bbox = [float(lon.min()) - 0.01, float(lat.min()) - 0.01, float(lon.max()) + 0.01, float(lat.max()) + 0.01]
r = request('POST', STAC, json=dict(collections=['sentinel-1-rtc'], bbox=bbox, datetime=f'{DR[0]}T00:00:00Z/{DR[1]}T00:00:00Z',
                                    limit=1000), timeout=120)
items = r.json()['features']
assert len(items) < 1000 and not any(l['rel'] == 'next' for l in r.json().get('links', []))
items = [it for it in items if pd.Timestamp(DR[0], tz='UTC') <= pd.Timestamp(it['properties']['datetime']) < pd.Timestamp(DR[1], tz='UTC')]
if not items:
    raise ValueError('No Sentinel-1 RTC scenes in the requested interval')
items = sorted(items, key=lambda it: (pd.Timestamp(it['properties']['datetime']).value, it['id']))


def cells(epsg, tf):
    """Raster row and column of every service pixel centre for a CRS and an affine transform (a, b, c, d, e, f)."""
    a, b, c, d, e, f = tf
    xs, ys = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True).transform(lon, lat)
    xs, ys = np.asarray(xs), np.asarray(ys)
    det = a * e - b * d
    return (np.floor((-d * (xs - c) + a * (ys - f)) / det).astype(np.int64),
            np.floor((e * (xs - c) - b * (ys - f)) / det).astype(np.int64))


def read(it):
    """Sample VV and VH at the service pixels, each with the georeferencing of its own file header.

    The STAC item-level proj:epsg / proj:transform can disagree with the files (seen for 2024/25 scenes: positive y scale,
    different origin), so they are recorded, not used. VV and VH of a scene must share CRS, transform and shape."""
    p = it['properties']
    geo = dict(item=dict(epsg=p.get('proj:epsg'), transform=[float(v) for v in p.get('proj:transform', [])[:6]],
                         shape=p.get('proj:shape')))
    if geo['item']['epsg'] is not None and len(geo['item']['transform']) == 6:
        ir, ic = cells(int(geo['item']['epsg']), geo['item']['transform'])
        geo['item'].update(rows=[int(ir.min()), int(ir.max())], cols=[int(ic.min()), int(ic.max())])
    out = np.full((len(lon), 2), NODATA, np.float32)
    for k, band in enumerate(('vv', 'vh')):
        asset = it['assets'][band]
        href = asset['href'] + '?' + token()
        for attempt in range(5):
            try:
                with rasterio.open(href) as s:
                    epsg, tf, shape = s.crs.to_epsg(), tuple(float(v) for v in tuple(s.transform)[:6]), (s.height, s.width)
                    rr, cc = cells(epsg, tf)
                    pad = int(np.ceil(PAD / abs(tf[0])))
                    r0, c0 = rr.min() - pad, cc.min() - pad
                    win = Window(c0, r0, cc.max() + pad + 1 - c0, rr.max() + pad + 1 - r0)
                    arr = s.read(1, window=win, boundless=True, fill_value=float(NODATA))
                    nodata = s.nodata
                break
            except rasterio.errors.RasterioIOError:
                if attempt == 4:
                    raise RuntimeError(f'{it["id"]}: read failed')                  # signed URL not logged
                time.sleep(5 * (attempt + 1))
        out[:, k] = arr[rr - r0, cc - c0]
        geo[band] = dict(epsg=epsg, transform=list(tf), shape=list(shape), nodata=None if nodata is None else float(nodata),
                         asset_proj={q: asset.get(q) for q in ('proj:epsg', 'proj:transform', 'proj:shape')},
                         rows=[int(rr.min()), int(rr.max())], cols=[int(cc.min()), int(cc.max())],
                         inside=bool(((rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])).all()))
    v, h = geo['vv'], geo['vh']
    if (v['epsg'], v['transform'], v['shape']) != (h['epsg'], h['transform'], h['shape']):
        raise ValueError(f'{it["id"]}: VV and VH grids differ')
    geo['item_matches_files'] = bool(geo['item']['epsg'] == v['epsg'] and geo['item']['transform'] == v['transform'])
    return dict(rtc_id=it['id'], time_ms=int(pd.Timestamp(p['datetime']).value // 1_000_000), orbit='A' if p['sat:orbit_state'] == 'ascending' else 'D',
                relative_orbit=int(p['sat:relative_orbit']), epsg=v['epsg'], platform=p['platform'], georef=geo), out


t0 = time.time()
token()
with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(read, items))
meta = pd.DataFrame([m for m, _ in res])
vals = np.stack([v for _, v in res])                                             # [images, pixels, 2]
valid = np.isfinite(vals) & (vals > 0)
has = (vals != NODATA).any(-1)                                                   # [images, pixels]
fi = pd.DataFrame(has.T).groupby(fid_px).sum().stack().rename('n_pixels_with_value').reset_index()
fi.columns = ['fid', 'image', 'n_pixels_with_value']
fi = fi[fi.n_pixels_with_value > 0].reset_index(drop=True)
save('RTC.npz', lambda f: np.savez(f, values=vals, valid=valid, time_ms=meta.time_ms.to_numpy(np.int64), orbit=meta.orbit.to_numpy().astype('U1'),
                                   rtc_id=meta.rtc_id.to_numpy().astype(str), epsg=meta.epsg.to_numpy()))
save('FIELD_IMAGES.parquet', lambda f: fi.to_parquet(f, index=False))
rep = dict(items=len(meta), orbits=meta.groupby(['orbit', 'relative_orbit']).size().rename('n').reset_index().to_dict('records'),
           first=str(pd.Timestamp(meta.time_ms.min(), unit='ms')), last=str(pd.Timestamp(meta.time_ms.max(), unit='ms')),
           pixels=len(lon), fields=int(len(fids)), valid_share_vv=float(valid[..., 0][has].mean()), valid_share_vh=float(valid[..., 1][has].mean()),
           images_per_field=fi.groupby('fid').size().describe().round(1).to_dict(), seconds=round(time.time() - t0, 1),
           source='MPC sentinel-1-rtc (STAC bbox search)', bbox=bbox,
           georeferencing='each file header (VV and VH checked equal); STAC item proj fields recorded only',
           item_proj_differs_from_files=int((~meta.georef.map(lambda g: g['item_matches_files'])).sum()),
           pixels_inside_every_file=bool(meta.georef.map(lambda g: g['vv']['inside'] and g['vh']['inside']).all()))
(OUT / 'IMAGES.json').write_text(json.dumps(dict(summary=rep, images=meta.to_dict('records')), indent=1) + '\n')
print(json.dumps(rep, indent=1))

