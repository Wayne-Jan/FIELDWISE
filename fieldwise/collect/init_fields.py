"""Start a collection workspace for one season from your fields: <root>/fields/* and <root>/collection.json.

Input folder: FIELDS.json (the 10 m grid: x0, y1, w, h, crs), FIELDS.parquet (one row per field: field, n_pix and the season
dates as UTC day numbers, either sow_day / harvest_day, or season_start_proxy_day / season_end_proxy_day from
season_proxies.py) and PIXELS.parquet (field, row, col, lon, lat of every service pixel). The collection steps read the
dates from sow_day / harvest_day; proxies are copied there and kept in their own columns with date_source.

usage: python -m fieldwise.collect.init_fields <fields_dir> <root> <sowing year>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

FD, ROOT = Path(sys.argv[1]), Path(sys.argv[2])
Y = int(sys.argv[3])                                                         # sowing year of the season
S = json.loads((FD / 'FIELDS.json').read_text())
g = {k: S['grid'][k] for k in ('x0', 'y1', 'w', 'h', 'crs')}
x0, y1 = g['x0'], g['y1']
x1, y0 = x0 + 10 * g['w'], y1 - 10 * g['h']
lon, lat = Transformer.from_crs(g['crs'], 'EPSG:4326', always_xy=True).transform([x0, x1, x0, x1], [y0, y0, y1, y1])
g['box'] = [round(min(lon), 5), round(min(lat), 5), round(max(lon), 5), round(max(lat), 5)]
cfg = dict(crop='corn', year=Y + 1, pre_start=f'{Y - 1}-01-01', weather_start=f'{Y}-01-01', base_start=f'{Y}-07-01',
           fm_end=f'{Y + 1}-06-01', end=f'{Y + 1}-09-01', grid=g,
           note=f'Season {Y}/{(Y + 1) % 100:02d} (sowing year {Y}). Interval ends are exclusive.')
F = pd.read_parquet(FD / 'FIELDS.parquet').sort_values('field').reset_index(drop=True)
if 'sow_day' not in F:                                                       # image-derived proxies, not sowing or harvest
    F['sow_day'], F['harvest_day'] = F.season_start_proxy_day, F.season_end_proxy_day
F['area_ha'] = F.n_pix * 0.01
px = pd.read_parquet(FD / 'PIXELS.parquet').sort_values(['field', 'row', 'col']).reset_index(drop=True)
ids = np.full((g['h'], g['w']), -1, np.int64)
ids[px.row, px.col] = px.field
out = ROOT / 'fields'
out.mkdir(parents=True, exist_ok=True)


def atomic(path, write):
    tmp = path.with_name('.' + path.name + '.tmp')
    write(tmp)
    os.replace(tmp, path)


atomic(out / 'FIELDS.parquet', lambda p: F.to_parquet(p, index=False))
atomic(out / 'PIXELS.parquet', lambda p: px[['field', 'row', 'col', 'lon', 'lat']].to_parquet(p, index=False))
atomic(out / 'field_id.npy', lambda p: np.save(open(p, 'wb'), ids))
atomic(out / 'FIELDS.json', lambda p: p.write_text(json.dumps(dict(grid=g, fields=len(F), pixels=len(px)), indent=2)))
atomic(ROOT / 'collection.json', lambda p: p.write_text(json.dumps(cfg, indent=1)))
print(json.dumps(dict(fields=len(F), pixels=len(px), box=g['box']), indent=1))
