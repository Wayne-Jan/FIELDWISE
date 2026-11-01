"""Public maize data workflow: extract_fm_products. Run through python -m fieldwise.collect; see docs/DATA_COLLECTION.md."""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ee
from .settings import config, project, weather_cells, month_edges
import numpy as np
import pandas as pd

FD, BI, OUT = map(Path, sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
ee.Initialize(project=project())
G = json.loads((FD / 'FIELDS.json').read_text())['grid']
GRID = {'dimensions': {'width': G['w'], 'height': G['h']},
        'affineTransform': {'scaleX': 10, 'shearX': 0, 'translateX': G['x0'], 'shearY': 0, 'scaleY': -10, 'translateY': G['y1']},
        'crsCode': G['crs']}
roi = ee.Geometry.Rectangle(G['box'])
with np.load(BI / 'corn_grid.npz') as z:
    ROW, COL = z['row'].astype(np.int64), z['col'].astype(np.int64)
S0, S1 = config()['pre_start'], config()['fm_end']
L2A = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'SCL']
L1C = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']
DW = ['label', 'water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice']
SPEC = {   # product: (collection, bands, date range, fill, extra filter)
    's2_l2a_pre': ('COPERNICUS/S2_SR_HARMONIZED', L2A, (config()['pre_start'], config()['base_start']), 0, None),
    's2_l1c': ('COPERNICUS/S2_HARMONIZED', L1C, (S0, S1), 0, None),
    'cloud_score_plus': ('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED', ['cs', 'cs_cdf'], (S0, S1), -1, None),
    'dynamic_world': ('GOOGLE/DYNAMICWORLD/V1', DW, (S0, S1), -1, None),
    's1_grd': ('COPERNICUS/S1_GRD', ['VV', 'VH', 'angle'], (S0, S1), -9999,
               lambda c: c.filter(ee.Filter.eq('instrumentMode', 'IW')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
               .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))),
    'viirs_dnb_monthly': ('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG', ['avg_rad', 'cf_cvg'], (str((pd.Timestamp(S0) - pd.offsets.MonthBegin(1)).date()), S1), -9999, None),
    'landscan_global': ('projects/sat-io/open-datasets/ORNL/LANDSCAN_GLOBAL', ['b1'], (S0, str(config()['year'] + 1) + '-01-01'), -1, None),
    'worldcereal_2021': ('ESA/WorldCereal/2021/MODELS/v100', ['classification'], None, -1, None),
}


def call(fn, what):
    for attempt in range(6):
        try:
            return fn()
        except ee.EEException as e:
            if attempt == 5:
                raise RuntimeError(f'{what}: {str(e)[:200]}')
            time.sleep(10 * (attempt + 1))


def listing(product):
    coll, bands, dr, fill, flt = SPEC[product]
    c = ee.ImageCollection(coll).filterBounds(roi)
    if dr:
        c = c.filterDate(*dr)
    if flt:
        c = flt(c)
    # system:time_end only where the product has an interval (VIIRS, LandScan); reduceColumns would drop images without it
    sel = ['system:index', 'system:time_start'] + (['system:time_end'] if product in ('viirs_dnb_monthly', 'landscan_global') else [])
    n = call(lambda: c.size().getInfo(), f'{product} size')
    info = call(lambda: c.reduceColumns(ee.Reducer.toList(len(sel)), sel).get('list').getInfo(), f'{product} listing')
    assert len(info) == n, (product, len(info), n)
    df = pd.DataFrame(info, columns=['key', 'start', 'end'][:len(sel)])
    return coll, bands, fill, df.assign(end=df.get('end', df.start))


def get(product, coll, bands, fill, key):
    out = OUT / 'parts' / product / f'{key}.npz'
    if out.exists():                                                      # safe to rerun
        return
    arr = call(lambda: ee.data.computePixels({'expression': ee.Image(f'{coll}/{key}').select(bands).unmask(fill), 'fileFormat': 'NUMPY_NDARRAY',
                                              'grid': GRID}), f'{product}/{key}')
    v = np.stack([np.asarray(arr[b])[ROW, COL] for b in bands], -1)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name('.tmp.' + out.name)
    with open(tmp, 'wb') as f:
        np.savez(f, v=v)
    os.replace(tmp, out)


log = {}
t0 = time.time()
for product in SPEC:
    coll, bands, fill, im = listing(product)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda k: get(product, coll, bands, fill, k), im.key))
    with ThreadPoolExecutor(8) as ex:                                     # [images, pixels, bands]
        V = np.stack(list(ex.map(lambda k: np.load(OUT / 'parts' / product / f'{k}.npz')['v'], im.key)))
    ok = V != fill
    keep = ok.any((1, 2))
    im, V, ok = im[keep].reset_index(drop=True), V[keep], ok[keep]
    if product == 'worldcereal_2021':
        s = pd.to_datetime(im.key.str.split('_').str[-2], format='%Y%m%d').to_numpy().astype('datetime64[ms]').astype(np.int64)
        e = pd.to_datetime(im.key.str.split('_').str[-1], format='%Y%m%d').to_numpy().astype('datetime64[ms]').astype(np.int64)
        acq, i0, i1 = s, s, e
    elif product in ('viirs_dnb_monthly', 'landscan_global'):
        acq, i0, i1 = im.start, im.start, im.end
    else:
        acq = i0 = i1 = im.start
    tmp = OUT / f'.tmp.{product}.npz'
    with open(tmp, 'wb') as f:
        np.savez(f, values=V.astype(np.float32) if V.dtype.kind == 'f' else V, found=ok, valid=ok.copy(), keys=im.key.to_numpy().astype(str),
                 acquisition_ms=np.asarray(acq, np.int64), interval_start_ms=np.asarray(i0, np.int64), interval_end_ms=np.asarray(i1, np.int64),
                 bands=np.array(bands))
    os.replace(tmp, OUT / f'{product}.npz')
    log[product] = dict(collection=coll, images=len(im), dropped_no_value=int((~keep).sum()), first=str(pd.Timestamp(int(np.min(acq)), unit='ms')),
                        last=str(pd.Timestamp(int(np.max(acq)), unit='ms')), valid_share=float(ok.mean()), seconds=round(time.time() - t0, 1))
    print(product, json.dumps(log[product]), flush=True)

# ---- TerraClimate at field representative points ----
F = pd.read_parquet(FD / 'FIELDS.parquet').assign(fid=lambda d: 900_000 + d.field)
pts = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {'fid': int(r.fid)}) for r in F.itertuples()])
tc = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE').filterDate(S0, S1).select(['aet', 'def', 'pdsi', 'ro', 'soil', 'swe'])
tl = call(lambda: tc.reduceColumns(ee.Reducer.toList(2), ['system:index', 'system:time_start']).get('list').getInfo(), 'terraclimate listing')


def tc_month(k):                                                          # one month per request (GEE 5000-element limit)
    key, start = tl[k]
    img = ee.Image(f'IDAHO_EPSCOR/TERRACLIMATE/{key}').select(['aet', 'def', 'pdsi', 'ro', 'soil', 'swe'])
    r = call(lambda: img.sampleRegions(pts, scale=4638.3, geometries=False).getInfo(), f'terraclimate {key}')
    end = (pd.Timestamp(int(start), unit='ms') + pd.offsets.MonthBegin(1)).value // 1_000_000
    return pd.DataFrame([f['properties'] for f in r['features']]).assign(image_id=key, time_start=int(start), time_end=int(end))


with ThreadPoolExecutor(6) as ex:
    T = pd.concat(list(ex.map(tc_month, range(len(tl)))), ignore_index=True)
T.to_parquet(OUT / '.tmp.terraclimate.parquet', index=False)
os.replace(OUT / '.tmp.terraclimate.parquet', OUT / 'terraclimate.parquet')
log['terraclimate'] = dict(rows=len(T), months=int(T.image_id.nunique()), fields=int(T.fid.nunique()))
print('terraclimate', log['terraclimate'], flush=True)

# ---- ERA5-Land hourly 2022, same nine cells ----
fc = ee.Geometry.MultiPoint(weather_cells(FD))
WX = ['temperature_2m', 'dewpoint_temperature_2m', 'total_precipitation_hourly', 'surface_solar_radiation_downwards_hourly',
      'surface_thermal_radiation_downwards_hourly', 'surface_net_solar_radiation_hourly', 'surface_net_thermal_radiation_hourly',
      'u_component_of_wind_10m', 'v_component_of_wind_10m', 'surface_pressure', 'potential_evaporation_hourly', 'total_evaporation_hourly',
      'evaporation_from_vegetation_transpiration_hourly', 'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2',
      'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4', 'skin_temperature', 'soil_temperature_level_1']
era = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').select(WX)
months = month_edges(config()['pre_start'], config()['weather_start'])


def month(k):
    r = call(lambda: era.filterDate(str(months[k].date()), str(months[k + 1].date())).getRegion(fc, 11131.949079327358).getInfo(), f'era5 {k}')
    return pd.DataFrame(r[1:], columns=r[0])


with ThreadPoolExecutor(4) as ex:
    wx = pd.concat(list(ex.map(month, range(len(months) - 1))), ignore_index=True)
wx['cell_lon'], wx['cell_lat'] = wx.longitude.round(1), wx.latitude.round(1)
wx.to_parquet(OUT / '.tmp.era5_land_pre.parquet', index=False)
os.replace(OUT / '.tmp.era5_land_pre.parquet', OUT / 'era5_land_pre.parquet')
log['era5_land_pre'] = dict(rows=len(wx), hours=int(wx.time.nunique()), cells=int(wx.groupby(['cell_lon', 'cell_lat']).ngroups))
log['seconds'] = round(time.time() - t0, 1)
(OUT / 'EXTRACT.json').write_text(json.dumps(log, indent=1) + '\n')
print(json.dumps(log, indent=1))

