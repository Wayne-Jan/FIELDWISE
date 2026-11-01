"""Reader for public acquisitions produced by fieldwise.collect. No YieldSAT loader or dataset is required."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from typing import NamedTuple

class Window(NamedTuple):
    start_ms: int
    end_ms: int
    decision_ms: int

DAY, HOUR = 86_400_000, 3_600_000
S2B = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'SCL']
WX = ['temperature_2m', 'dewpoint_temperature_2m', 'total_precipitation_hourly', 'surface_solar_radiation_downwards_hourly',
      'surface_thermal_radiation_downwards_hourly', 'surface_net_solar_radiation_hourly', 'surface_net_thermal_radiation_hourly',
      'u_component_of_wind_10m', 'v_component_of_wind_10m', 'surface_pressure', 'potential_evaporation_hourly', 'total_evaporation_hourly',
      'evaporation_from_vegetation_transpiration_hourly', 'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2',
      'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4', 'skin_temperature', 'soil_temperature_level_1']
TC_BANDS = ['aet', 'def', 'pdsi', 'ro', 'soil', 'swe']
TC_SCALE = dict(aet=0.1, pdsi=0.01, ro=1.0, soil=0.1, swe=1.0, **{'def': 0.1})
PRODUCTS = ('s1_grd', 's2_l1c', 'cloud_score_plus', 'dynamic_world', 'viirs_dnb_monthly', 'landscan_global', 'worldcereal_2021')


def corners(lon, lat):
    """build_g_inputs.corners: bilinear weights of the four 0.1-degree cells around the point"""
    lo0, la0 = np.floor(lon * 10) / 10, np.floor(lat * 10) / 10
    tx, ty = (lon - lo0) * 10, (lat - la0) * 10
    return [((round(lo0, 1), round(la0, 1)), (1 - tx) * (1 - ty)), ((round(lo0 + .1, 1), round(la0, 1)), tx * (1 - ty)),
            ((round(lo0, 1), round(la0 + .1, 1)), (1 - tx) * ty), ((round(lo0 + .1, 1), round(la0 + .1, 1)), tx * ty)]


class CollectionData:
    def __init__(self, tw):
        tw = Path(tw)
        self.cases = pd.read_parquet(tw / 'base_inputs/CASES.parquet').set_index('case_id')
        with np.load(tw / 'base_inputs/corn_grid.npz') as z:
            row, col, fids, pix0 = z['row'].astype(np.int64), z['col'].astype(np.int64), z['fid'], z['pix0']
        self.fid_px = np.repeat(fids, np.diff(pix0))
        with np.load(tw / 'raw/s1_rtc/PIXELS_LONLAT.npz') as z:
            self.lon, self.lat = z['lon'], z['lat']
            assert np.array_equal(z['fid'], self.fid_px)
        self.sl = {int(f): slice(int(pix0[k]), int(pix0[k + 1])) for k, f in enumerate(fids)}
        F = pd.read_parquet(tw / 'fields/FIELDS.parquet').assign(fid=lambda d: 900_000 + d.field).set_index('fid')
        self.fields = F
        # ---- S2 L2A: base grids (2023-07 on) + pre-season service-pixel values ----
        im = pd.DataFrame(json.loads((tw / 'raw/base_inputs/s2/IMAGES.json').read_text()))[['image_id', 'time_ms', 'tile']]
        cube = np.zeros((len(im), len(row), len(S2B)), np.uint16)

        def read(i):
            with np.load(tw / f'raw/base_inputs/s2/{im.image_id[i]}.npz') as z:
                cube[i] = np.stack([z[b][row, col] for b in S2B], -1)
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(read, range(len(im))))
        with np.load(tw / 'raw/fm_products/s2_l2a_pre.npz') as z:
            assert list(z['bands']) == S2B
            pre = pd.DataFrame(dict(image_id=z['keys'], time_ms=z['acquisition_ms'], tile=[k[-5:] for k in z['keys']]))
            cube = np.concatenate([z['values'].astype(np.uint16), cube])
        im = pd.concat([pre, im], ignore_index=True)
        assert im.image_id.is_unique
        o = np.lexsort((im.tile.to_numpy(), im.time_ms.to_numpy()))
        self.s2_im, self.s2_cube = im.iloc[o].reset_index(drop=True), cube[o]
        scl_any = self.s2_cube[..., 12] != 0
        self.s2_of = {f: np.flatnonzero(scl_any[:, s].any(1)) for f, s in self.sl.items()}
        # ---- S1 RTC: both extractions ----
        parts, fis, off = [], [], 0
        for d in ('raw/s1_rtc_pre', 'raw/s1_rtc'):
            with np.load(tw / d / 'RTC.npz') as z:
                parts.append({k: z[k] for k in z.files})
            fis.append(pd.read_parquet(tw / d / 'FIELD_IMAGES.parquet').assign(image=lambda g, o_=off: g.image + o_))
            off += len(parts[-1]['time_ms'])
        self.r = {k: np.concatenate([p[k] for p in parts]) for k in ('values', 'valid', 'time_ms', 'orbit', 'rtc_id')}
        assert len(np.unique(self.r['rtc_id'])) == len(self.r['rtc_id'])
        fi = pd.concat(fis, ignore_index=True)
        rank = np.empty(len(self.r['time_ms']), np.int64)
        rank[np.lexsort((self.r['rtc_id'], self.r['time_ms']))] = np.arange(len(rank))
        self.r1_of = {int(f): g.image.to_numpy()[np.argsort(rank[g.image.to_numpy()])] for f, g in fi.groupby('fid')}
        # ---- image products ----
        self.img = {}
        for p in PRODUCTS:
            with np.load(tw / f'raw/fm_products/{p}.npz') as z:
                d = {k: z[k] for k in z.files}
            d['bands'] = [str(b) for b in d['bands']]
            fo = d['found'].any(2)                                                       # [images, pixels]
            d['of'] = {f: np.flatnonzero(fo[:, s].any(1)) for f, s in self.sl.items()}
            self.img[p] = d
        # ---- ERA5-Land cells ----
        wx = pd.concat([pd.read_parquet(tw / 'raw/fm_products/era5_land_pre.parquet'), pd.read_parquet(tw / 'raw/base_inputs/era5/ERA5_HOURLY.parquet')],
                       ignore_index=True)
        self.cell = {}
        for (lo, la), g in wx.groupby(['cell_lon', 'cell_lat']):
            g = g.sort_values('time')
            assert g.time.is_unique
            self.cell[(round(lo, 1), round(la, 1))] = (g.time.to_numpy(np.int64), g[WX].to_numpy(np.float64))
        self.flux = np.array([b.endswith('_hourly') for b in WX])
        # ---- TerraClimate, terrain ----
        self.tc = pd.read_parquet(tw / 'raw/fm_products/terraclimate.parquet')
        with np.load(tw / 'terrain/TERRAIN.npz') as z:                               # own pixel order: matched by (row, col)
            k = pd.Index(z['row'].astype(np.int64) * 100_000 + z['col']).get_indexer(row * 100_000 + col)
            assert (k >= 0).all() and len(np.unique(k)) == len(k)
            self.elev = z['elevation_m'].astype(np.float64)[k]
            self.rise = np.tan(np.radians(z['slope_deg'].astype(np.float64)))[k]
        self.manifest = {'meta': {'products': {'terraclimate': {'scalars': {'bands': TC_BANDS}}}}}

    # ------------------------------------------------------------------ cases
    def case(self, case_id):
        if case_id.endswith(':pre_sowing_calendar_year'):
            tau1 = case_id.replace(':pre_sowing_calendar_year', ':tau1')
            r = self.cases.loc[tau1]
            y = pd.Timestamp(int(r.input_start_utc_ms), unit='ms').year
            s, e = (int(pd.Timestamp(f'{y - 1}-01-01').value // 1_000_000), int(pd.Timestamp(f'{y}-01-01').value // 1_000_000))
            return int(r.fid), Window(s, e, e + 6 * DAY)
        r = self.cases.loc[case_id]
        return int(r.fid), Window(int(r.input_start_utc_ms), int(r.window_end_utc_ms), int(r.decision_utc_ms))

    def pixels(self, fid):
        s = self.sl[int(fid)]
        return dict(fid=int(fid), lon=self.lon[s], lat=self.lat[s])

    def _pos(self, fid, positions):
        s = self.sl[int(fid)]
        return s, (np.arange(s.stop - s.start) if positions is None else np.asarray(positions, np.int64))

    # ------------------------------------------------------------------ S2 L2A, S1 RTC
    def s2(self, fid, w, positions=None):
        s, pos = self._pos(fid, positions)
        idx = self.s2_of[int(fid)]
        t = self.s2_im.time_ms.to_numpy(np.int64)[idx]
        idx = idx[(t >= w.start_ms) & (t < w.end_ms) & (t + 24 * HOUR <= w.decision_ms)]
        v = self.s2_cube[idx][:, s.start + pos].astype(np.float64)
        found = v != 0
        sel = self.s2_im.loc[idx]
        return dict(fid=int(fid), image_ids=sel.image_id.tolist(), kind=np.full(len(idx), 'gee_s2_sr_harmonized', object),
                    acquisition_ms=sel.time_ms.to_numpy(np.int64), boa_add_offset=np.zeros(len(idx), np.int64), bands=S2B, positions=pos,
                    values=np.where(found, v, np.nan), found=found, valid=found.copy(), window=w._asdict())

    def s1_rtc(self, fid, w=None, *, latency_hours=48, positions=None):
        s, pos = self._pos(fid, positions)
        idx = self.r1_of.get(int(fid), np.zeros(0, np.int64))
        t = self.r['time_ms'][idx]
        if w is not None:
            idx = idx[(t >= w.start_ms) & (t < w.end_ms) & (t + latency_hours * HOUR <= w.decision_ms)]
        values = self.r['values'][idx][:, s.start + pos]
        return dict(product='s1_rtc', units='gamma0_linear_power', fid=int(fid), bands=['VV', 'VH'], positions=pos, values=values,
                    found=np.ones(values.shape, bool), valid=self.r['valid'][idx][:, s.start + pos], acquisition_ms=self.r['time_ms'][idx],
                    keys=self.r['rtc_id'][idx].tolist(), orbit=self.r['orbit'][idx].astype(object), unavailable=[],
                    served_by='taiwan_mpc_rtc', latency_hours=latency_hours)

    # ------------------------------------------------------------------ image products
    def image(self, product, fid, w=None, *, latency_hours=None, positions=None):
        d = self.img[product]
        s, pos = self._pos(fid, positions)
        idx = d['of'][int(fid)]
        if w is not None:
            if product == 'worldcereal_2021':
                raise ValueError('worldcereal_2021 is archive-only')
            if latency_hours is None:
                raise ValueError(f'{product}: latency_hours must be given')
            a, e = d['interval_start_ms'][idx], d['interval_end_ms'][idx]
            inst = a == e
            idx = idx[np.where(inst, (a >= w.start_ms) & (a < w.end_ms), (a >= w.start_ms) & (e <= w.end_ms)) & (e + int(round(latency_hours * HOUR)) <= w.decision_ms)]
        fo = d['found'][idx][:, s.start + pos]
        v = d['values'][idx][:, s.start + pos].astype(np.float64)
        return dict(product=product, fid=int(fid), keys=d['keys'][idx].tolist(), acquisition_ms=d['acquisition_ms'][idx],
                    interval_start_ms=d['interval_start_ms'][idx], interval_end_ms=d['interval_end_ms'][idx], bands=d['bands'], positions=pos,
                    values=np.where(fo, v, np.nan), found=fo, valid=d['valid'][idx][:, s.start + pos],
                    window=None if w is None else w._asdict(), latency_hours=latency_hours)

    # ------------------------------------------------------------------ weather, TerraClimate, terrain
    def weather(self, product, fid, w):
        assert product == 'era5_land'
        f = self.fields.loc[int(fid)]
        cw = corners(float(f.lon), float(f.lat))
        lat_ms = 6 * DAY
        raws, times, inws, off = [], [], [], [0]
        for k, _ in cw:
            t, v = self.cell[k]
            a = int(np.searchsorted(t, w.start_ms, 'left'))
            b = max(a, int(np.searchsorted(t, min(w.end_ms, w.decision_ms - lat_ms), 'right')))
            t, v = t[a:b], v[a:b]
            tt = t[:, None]
            inb = np.where(self.flux[None, :], (tt - HOUR >= w.start_ms) & (tt <= w.end_ms), (tt >= w.start_ms) & (tt < w.end_ms)) & (tt + lat_ms <= w.decision_ms)
            keep = inb.any(1)
            raws.append(v[keep]), times.append(t[keep]), inws.append(inb[keep]), off.append(off[-1] + int(keep.sum()))
        values = np.concatenate(raws)
        valid = np.isfinite(values) & np.concatenate(inws)
        fill = values <= -1e14
        values[fill] = np.nan
        valid &= ~fill
        j = WX.index('total_precipitation_hourly')
        p = values[:, j]
        zero, bad = (p < 0) & (p >= -1e-6), p < -1e-6
        values[zero, j] = 0.0
        values[bad, j] = np.nan
        valid[bad, j] = False
        return dict(product=product, fid=int(fid), bands=WX, time_ms=np.concatenate(times), cell_offsets=np.asarray(off),
                    cells=np.array([f'{k[0]}_{k[1]}' for k, _ in cw]), weights=np.array([wt for _, wt in cw], np.float64), values=values, valid=valid,
                    window=w._asdict())

    def product(self, name, fid):
        assert name == 'terraclimate'
        g = self.tc[self.tc.fid == int(fid)].sort_values('time_start')
        raw = g[TC_BANDS].to_numpy(np.float64)
        return dict(product=name, fid=int(fid), bands=TC_BANDS, values=raw * np.array([TC_SCALE[b] for b in TC_BANDS]),
                    valid=np.isfinite(raw), quality_valid=np.ones(raw.shape, bool), time_start_ms=g.time_start.to_numpy(np.int64),
                    time_end_ms=g.time_end.to_numpy(np.int64), image_id=g.image_id.to_numpy())

    def static(self, fid):
        s = self.sl[int(fid)]
        return dict(terrain_elevation_m=self.elev[s], terrain_slope_rise=self.rise[s])
