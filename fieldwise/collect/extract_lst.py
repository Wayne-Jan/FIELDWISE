"""Public maize data workflow: extract_lst. Run through python -m fieldwise.collect; see docs/DATA_COLLECTION.md."""
import json
import os
import sys
from pathlib import Path

import ee
from .settings import config, project, weather_cells, month_edges
import numpy as np
import pandas as pd

FD, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
R, SZ, X0, Y0 = 6371007.181, 926.625433055833, -20015109.354, 10007554.677
px = pd.read_parquet(FD / 'PIXELS.parquet')
lam, phi = np.radians(px.lon.to_numpy()), np.radians(px.lat.to_numpy())
x, y = R * lam * np.cos(phi), R * phi
px['cell_col'] = np.floor((x - X0) / SZ).astype(int)
px['cell_row'] = np.floor((Y0 - y) / SZ).astype(int)
cells = px.groupby(['cell_row', 'cell_col']).size().rename('n_pix').reset_index()
cy = Y0 - (cells.cell_row + 0.5) * SZ
cx = X0 + (cells.cell_col + 0.5) * SZ
cells['lat'] = np.degrees(cy / R)
cells['lon'] = np.degrees(cx / (R * np.cos(np.radians(cells.lat))))
ee.Initialize(project=project())
col = ee.ImageCollection('MODIS/061/MOD11A1').filterDate(config()['base_start'], config()['end']).select(['LST_Day_1km', 'LST_Night_1km', 'QC_Day', 'QC_Night'])
pts = ee.Geometry.MultiPoint(cells[['lon', 'lat']].to_numpy().tolist())
proj = col.first().projection()
parts = []
for a, b in zip(month_edges(config()['base_start'], config()['end'])[:-1], month_edges(config()['base_start'], config()['end'])[1:]):     # two half-year requests
    r = col.filterDate(str(a.date()), str(b.date())).getRegion(pts, SZ, proj).getInfo()
    parts.append(pd.DataFrame(r[1:], columns=r[0]))
d = pd.concat(parts, ignore_index=True)
lam2, phi2 = np.radians(d.longitude.to_numpy(float)), np.radians(d.latitude.to_numpy(float))
d['cell_col'] = np.floor((R * lam2 * np.cos(phi2) - X0) / SZ).astype(int)
d['cell_row'] = np.floor((Y0 - R * phi2) / SZ).astype(int)
d['day'] = (d.time.astype('int64') // 86_400_000).astype(int)
d = d[['cell_row', 'cell_col', 'day', 'LST_Day_1km', 'LST_Night_1km', 'QC_Day', 'QC_Night']].fillna(0)
for c in ('LST_Day_1km', 'LST_Night_1km', 'QC_Day', 'QC_Night'):
    d[c] = d[c].astype(np.int64)
got = set(map(tuple, d[['cell_row', 'cell_col']].drop_duplicates().to_numpy()))
want = set(map(tuple, cells[['cell_row', 'cell_col']].to_numpy()))
assert got == want, ('cells returned by GEE differ from the service-pixel cells', got ^ want)
for name, obj in (('LST_CELLS.parquet', cells), ('PIXEL_CELL.parquet', px[['field', 'row', 'col', 'cell_row', 'cell_col']]),
                  ('LST_DAILY.parquet', d)):
    tmp = OUT / f'.tmp.{name}'
    obj.to_parquet(tmp, index=False)
    os.replace(tmp, OUT / name)
print(json.dumps(dict(cells=len(cells), days=int(d.day.nunique()), rows=len(d), day_present_share=float((d.LST_Day_1km != 0).mean()),
                      night_present_share=float((d.LST_Night_1km != 0).mean()))))

