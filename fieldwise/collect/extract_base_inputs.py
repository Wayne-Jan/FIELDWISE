"""Public maize data workflow: extract_base_inputs. Run through python -m fieldwise.collect; see docs/DATA_COLLECTION.md."""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ee
from .settings import config, project, weather_cells, month_edges
import numpy as np
import pandas as pd

FD, OUT = Path(sys.argv[1]), Path(sys.argv[2])
G = json.loads((FD / 'FIELDS.json').read_text())['grid']
ee.Initialize(project=project())
S2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'SCL']
WX_BANDS = ['temperature_2m', 'dewpoint_temperature_2m', 'total_precipitation_hourly', 'surface_solar_radiation_downwards_hourly',
            'surface_thermal_radiation_downwards_hourly', 'surface_net_solar_radiation_hourly', 'surface_net_thermal_radiation_hourly',
            'u_component_of_wind_10m', 'v_component_of_wind_10m', 'surface_pressure', 'potential_evaporation_hourly',
            'total_evaporation_hourly', 'evaporation_from_vegetation_transpiration_hourly', 'volumetric_soil_water_layer_1',
            'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4', 'skin_temperature',
            'soil_temperature_level_1']
SOIL_PROPS = ('cec', 'cfvo', 'clay', 'nitrogen', 'phh2o', 'sand', 'silt', 'soc')
SOIL_DEPTHS = ('0-5', '5-15', '15-30')
GRID = {'dimensions': {'width': G['w'], 'height': G['h']},
        'affineTransform': {'scaleX': 10, 'shearX': 0, 'translateX': G['x0'], 'shearY': 0, 'scaleY': -10, 'translateY': G['y1']},
        'crsCode': G['crs']}
lon0, lat0, lon1, lat1 = G['box']
roi = ee.Geometry.Rectangle([lon0, lat0, lon1, lat1])


def write_npz(path, **arrs):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def pixels(img, bands, grid=GRID):
    arr = ee.data.computePixels({'expression': img.select(bands), 'fileFormat': 'NUMPY_NDARRAY', 'grid': grid})
    return {b: np.asarray(arr[b]) for b in bands}


log = {}
# ---- S2 (about 80 images; 6 parallel requests) ----
col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).filterDate(config()['base_start'], config()['end'])
ids = col.aggregate_array('system:index').getInfo()
times = col.aggregate_array('system:time_start').getInfo()
bl = col.aggregate_array('PROCESSING_BASELINE').getInfo()
tiles = col.aggregate_array('MGRS_TILE').getInfo()


def get_s2(k):
    out = OUT / 's2' / f'{ids[k]}.npz'
    if not out.exists():                                     # safe to rerun
        px = pixels(ee.Image(f'COPERNICUS/S2_SR_HARMONIZED/{ids[k]}').unmask(0), S2_BANDS)
        write_npz(out, **{b: px[b].astype(np.uint16) for b in S2_BANDS})
    return k


with ThreadPoolExecutor(6) as ex:
    done = list(ex.map(get_s2, range(len(ids))))
imgs = pd.DataFrame(dict(image_id=ids, time_ms=times, baseline=bl, tile=tiles)).sort_values(['time_ms', 'tile'])
(OUT / 's2').mkdir(parents=True, exist_ok=True)
(OUT / 's2' / 'IMAGES.json').write_text(imgs.to_json(orient='records'))
log['s2'] = dict(images=len(ids), first=str(np.datetime64(min(times), 'ms')), last=str(np.datetime64(max(times), 'ms')),
                 baselines=sorted(set(bl)), tiles=imgs.tile.value_counts().to_dict())
print('s2', log['s2'], flush=True)

# ---- ERA5-Land hourly at 6 native cell centres, month by month ----
pts = weather_cells(FD)
fc = ee.Geometry.MultiPoint([list(p) for p in pts])
months = month_edges(config()['weather_start'], config()['end'])
rows = []
era = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').select(WX_BANDS)


def get_month(k):
    r = era.filterDate(str(months[k].date()), str(months[k + 1].date())).getRegion(fc, 11131.949079327358).getInfo()
    return pd.DataFrame(r[1:], columns=r[0])


with ThreadPoolExecutor(4) as ex:
    parts = list(ex.map(get_month, range(len(months) - 1)))
wx = pd.concat(parts, ignore_index=True)
wx['cell_lon'] = wx.longitude.round(1)
wx['cell_lat'] = wx.latitude.round(1)
(OUT / 'era5').mkdir(parents=True, exist_ok=True)
tmp = OUT / 'era5' / 'ERA5_HOURLY.tmp.parquet'
wx.to_parquet(tmp, index=False)
os.replace(tmp, OUT / 'era5' / 'ERA5_HOURLY.parquet')
log['era5'] = dict(rows=len(wx), cells=sorted({(a, b) for a, b in zip(wx.cell_lon, wx.cell_lat)}),
                   null_share={b: float(wx[b].isna().mean()) for b in WX_BANDS[:3]}, hours=int(wx.time.nunique()))
print('era5', log['era5'], flush=True)

# SoilGrids is collected exclusively with soil_official.py (ISRIC).

# ---- SRTM 1 arcsecond, box + 20 km ----
buf = 20 / 111.32
b0 = [lon0 - buf / np.cos(np.radians(lat0)), lat0 - buf, lon1 + buf / np.cos(np.radians(lat1)), lat1 + buf]
step = 1 / 3600
w = int(np.ceil((b0[2] - b0[0]) / step))
h = int(np.ceil((b0[3] - b0[1]) / step))
x0 = np.floor(b0[0] / step) * step - step / 2                 # SRTMGL1 cell edges sit half an arcsecond off the integer grid
y1 = np.ceil(b0[3] / step) * step + step / 2
srtm_grid = {'dimensions': {'width': w, 'height': h},
             'affineTransform': {'scaleX': step, 'shearX': 0, 'translateX': x0, 'shearY': 0, 'scaleY': -step, 'translateY': y1},
             'crsCode': 'EPSG:4326'}
tiles = []
for r0 in range(0, h, 1000):                                    # requests of at most 1000 rows
    g = dict(srtm_grid, dimensions={'width': w, 'height': min(1000, h - r0)},
             affineTransform=dict(srtm_grid['affineTransform'], translateY=y1 - r0 * step))
    tiles.append(pixels(ee.Image('USGS/SRTMGL1_003').unmask(-32768), ['elevation'], g)['elevation'])
dem = np.concatenate(tiles).astype(np.int16)
write_npz(OUT / 'srtm' / 'SRTM.npz', elevation=dem, x0=x0, y1=y1, step=step)
log['srtm'] = dict(shape=list(dem.shape), nodata_share=float((dem == -32768).mean()), min=int(dem[dem != -32768].min()),
                   max=int(dem.max()), x0=x0, y1=y1)
print('srtm', log['srtm'], flush=True)
tmp = OUT / 'EXTRACT.tmp.json'
tmp.write_text(json.dumps(log, indent=1, default=str))
os.replace(tmp, OUT / 'EXTRACT.json')

