"""Sentinel-2 L2A raw pixels (B4, B8, SCL) of one season over a sub-box of the Tainan 10 m grid, for the season-date proxy
and the candidate areas (example tool, see season_proxies.py).
GEE values only (computePixels), NDVI and phenology computed offline. One npz per image (skipped when
present and readable); written as .tmp then renamed.

usage: python -m fieldwise.collect.extract_s2_season <out_dir> <x0> <y1> <w> <h> <start> <end>   (EPSG:32651, 10 m, ends exclusive)
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ee
import numpy as np

PROJECT = os.environ['FIELDWISE_EE_PROJECT']                              # your own Earth Engine project
BANDS = ['B4', 'B8', 'SCL']
OUT = Path(sys.argv[1])
G = dict(x0=float(sys.argv[2]), y1=float(sys.argv[3]), w=int(sys.argv[4]), h=int(sys.argv[5]))
START, END = sys.argv[6], sys.argv[7]


def fetch(img_id):
    dst = OUT / f"{img_id.split('/')[-1]}.npz"
    if dst.exists():
        try:
            np.load(dst)['B4']
            return 'skip'
        except Exception:
            dst.unlink()
    req = {'expression': ee.Image(img_id).select(BANDS), 'fileFormat': 'NUMPY_NDARRAY',
           'grid': {'dimensions': {'width': G['w'], 'height': G['h']},
                    'affineTransform': {'scaleX': 10, 'shearX': 0, 'translateX': G['x0'],
                                        'shearY': 0, 'scaleY': -10, 'translateY': G['y1']},
                    'crsCode': 'EPSG:32651'}}
    for attempt in range(5):
        try:
            arr = ee.data.computePixels(req)
            break
        except Exception as e:
            if attempt == 4:
                return f'fail {str(e)[:100]}'
            time.sleep(5 * (attempt + 1))
    tmp = dst.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **{b: np.asarray(arr[b]) for b in BANDS})
    os.replace(tmp, dst)
    return 'ok'


def main():
    ee.Initialize(project=PROJECT)
    OUT.mkdir(parents=True, exist_ok=True)
    x1, y0 = G['x0'] + 10 * G['w'], G['y1'] - 10 * G['h']
    poly = ee.Geometry.Rectangle([G['x0'], y0, x1, G['y1']], 'EPSG:32651', False)
    col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(poly).filterDate(START, END)
    meta = col.reduceColumns(ee.Reducer.toList(4), ['system:index', 'system:time_start', 'CLOUDY_PIXEL_PERCENTAGE',
                                                    'MGRS_TILE']).get('list').getInfo()
    ids = [f'COPERNICUS/S2_SR_HARMONIZED/{m[0]}' for m in meta]
    (OUT / 'GRID.json').write_text(json.dumps(dict(G, crs='EPSG:32651', start=START, end=END, bands=BANDS, project=PROJECT,
                                                   collection='COPERNICUS/S2_SR_HARMONIZED'), indent=1))
    (OUT / 'IMAGES.json').write_text(json.dumps(
        [dict(id=i, time_ms=m[1], cloudy_pct=m[2], tile=m[3]) for i, m in zip(ids, meta)], indent=1))
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(fetch, ids))
    status = dict(n=len(ids), results={s: res.count(s) for s in set(res)}, seconds=round(time.time() - t0, 1),
                  failed=[i for i, r in zip(ids, res) if r.startswith('fail')])
    (OUT / 'STATUS.json').write_text(json.dumps(status, indent=1))
    print(json.dumps(status))


if __name__ == '__main__':
    main()
