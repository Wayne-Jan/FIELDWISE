"""Public maize data workflow: soil_official. Run through python -m fieldwise.collect; see docs/DATA_COLLECTION.md."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
os.environ.setdefault('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.vrt,.tif')
os.environ.setdefault('GDAL_HTTP_MAX_RETRY', '5')
os.environ.setdefault('GDAL_HTTP_RETRY_DELAY', '3')
from .settings import config, project, weather_cells, month_edges
import numpy as np
import rasterio
from rasterio.warp import transform
from rasterio.windows import Window

FD, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
G = json.loads((FD / 'FIELDS.json').read_text())['grid']
BASE = '/vsicurl/https://files.isric.org/soilgrids/latest/data'
VARS = ('cec', 'cfvo', 'clay', 'nitrogen', 'phh2o', 'sand', 'silt', 'soc')
DEPTHS = ('0-5', '5-15', '15-30')
h, w = G['h'], G['w']
cy, cx = np.mgrid[0:h, 0:w]
x = (G['x0'] + (cx.ravel() + 0.5) * 10).tolist()
y = (G['y1'] - (cy.ravel() + 0.5) * 10).tolist()
out = {}
with rasterio.open(f'{BASE}/clay/clay_0-5cm_mean.vrt') as ds:
    hx, hy = transform(G['crs'], ds.crs, x, y)
    r, c = rasterio.transform.rowcol(ds.transform, hx, hy)
    r, c = np.asarray(r), np.asarray(c)
    r0, r1, c0, c1 = r.min(), r.max() + 1, c.min(), c.max() + 1
    ref_tr, ref_shape = ds.transform, ds.shape
for v in VARS:                                                  # 24 layers, one small window each
    for d in DEPTHS:
        with rasterio.open(f'{BASE}/{v}/{v}_{d}cm_mean.vrt') as ds:
            if ds.transform == ref_tr and ds.shape == ref_shape:
                rr, cc, a0, a1, b0, b1 = r, c, r0, r1, c0, c1
            else:                                               # e.g. soc: a slightly different grid; map with its own transform
                rr, cc = (np.asarray(t) for t in rasterio.transform.rowcol(ds.transform, hx, hy))
                a0, a1, b0, b1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
            win = ds.read(1, window=Window(b0, a0, b1 - b0, a1 - a0)).astype(np.int32)
        out[f'{v}_{d}cm_mean'] = win[rr - a0, cc - b0].reshape(h, w)
        print(v, d, int(np.median(out[f'{v}_{d}cm_mean'])), flush=True)
tmp = OUT / '.tmp.SOIL.npz'
np.savez_compressed(tmp, **out)
os.replace(tmp, OUT / 'SOIL.npz')
(OUT / 'SOURCE.json').write_text(json.dumps(dict(source=BASE, window_rows=[int(r0), int(r1)], window_cols=[int(c0), int(c1)],
                                                 nodata_share={k: float((a == -32768).mean()) for k, a in out.items()}), indent=1))

