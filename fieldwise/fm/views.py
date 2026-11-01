"""Foundation-model inputs of one field and observation window, in each model's official form.

For one field, one window (start, end E, decision C) and a chunk of service pixels, builds the input of each foundation model
(bands, units, time steps, compositing, group masks) before the official normalisation. Data come through a reader D with
the methods pixels, s2, s1_rtc, image, weather, product, static and manifest (see fieldwise/fm/README.md; the Taiwan
example implements it). Official sources (checked 2026-10-10):

  TESSERA v2 student  vendored tessera_infer_v2.encode_pixels: every clear S2 L2A acquisition (harmonised DN, SCL not in
                      {0,1,2,3,8,9}, 10 bands) with its day of year; S1 ascending and descending kept apart, official int16 scale
                      trunc(clip((20 log10(power) + 50) x 200, 0, 32767)), using native float32 operations.
  Galileo             nasaharvest/galileo src/data/earthengine + config: 30-day steps; S1 GRD VV VH (dB) median of the images
                      nearest the step midpoint (within max(15 d, nearest)), one orbit pass; S2_HARMONIZED (L1C) quality mosaic,
                      10 bands; NDVI of the mosaic; ERA5-Land temperature_2m (K) and total_precipitation_sum (m per month);
                      TerraClimate def soil aet; VIIRS avg_rad; SRTM elevation (m) and slope (deg); Dynamic World class
                      probabilities averaged over the window; WorldCereal 2021 classification (0 / 100, missing 0); LandScan b1
                      (missing 0); location x y z; static DW / WC = field mean (official: tile mean).
  Presto              nasaharvest/presto (openmapflow 0.2.2): 30-day steps; S1 as Galileo; COPERNICUS/S2 (L1C) quality mosaic of
                      13 bands (B1 / B10 removed, B9 dropped by the official normalize); ERA5_LAND/MONTHLY temperature_2m (K) and
                      total_precipitation as the mean daily total (m / day; official max 0.03); SRTM; Dynamic World label mode per
                      step (9 = missing); pixel latitude / longitude; month of the first step.
  OlmoEarth v1_2      olmoearth_pretrain + rslearn OlmoEarthEmbeddings: 30-day periods; sentinel2_l2a harmonised 12 bands, scenes
                      in cloud order, first clear acquisition per pixel; sentinel1 vv vh (dB) first acquisition of the period;
                      timestamps (day, month0, year) of the period start; Landsat not in the dataset (masked).

Substitutions of the official recipe (the official intermediates are not stored): Cloud Score+ cs_cdf replaces the official TOA
cloud / shadow score as the quality-mosaic key; TESSERA requires RTC gamma0 (no GRD fallback); ERA5-Land hourly aggregated over
each 30-day step replaces the monthly ERA5-Land images; the OlmoEarth scene order uses the field's clear fraction instead of the
scene cloud percentage; the last step is truncated at the window end (Galileo's exporter drops a partial last step). The reader
holds TerraClimate physical values; x 10 gives the official stored integers.

Time rule: every observation comes through the reader's window (start <= t < E, available <= C) with the latency proxies of
LATENCY_H. Monthly and annual background products (VIIRS, LandScan, WorldCereal 2021, TerraClimate) are taken from images whose
interval ends by E (VIIRS, TerraClimate) or the nearest year (LandScan, WorldCereal 2021); their release times are not applied.
Each view lists these relaxations in `relaxed`.
"""
from __future__ import annotations

import warnings

import numpy as np

DAY, HOUR = 86_400_000, 3_600_000
PERIOD_MS = 30 * DAY
LATENCY_H = dict(s1_grd=48, s2_l1c=24, cloud_score_plus=24, dynamic_world=48)   # research proxies (README); S1: PRODUCT_RULES 2-day
L2A = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']   # D.s2 band order (SCL is band 12)
SCL_INVALID_TESSERA = (0, 1, 2, 3, 8, 9)
SCL_CLEAR = (4, 5)
TESSERA_S2 = ('B4', 'B2', 'B3', 'B8', 'B8A', 'B5', 'B6', 'B7', 'B11', 'B12')
OLMO_S2 = ('B2', 'B3', 'B4', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12', 'B1', 'B9')
GALILEO_S2 = ('B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12')
PRESTO_S2 = ('B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12')    # official BANDS order (B1, B10 removed)
DW_CLASSES = ('water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice')
WC_PRODUCTS = ('temporarycrops', 'maize', 'wintercereals', 'springcereals', 'irrigation')
TC_BANDS = ('def', 'soil', 'aet')
OLMO_MISSING = -99999.0
GALILEO_ST_GROUPS = ([0, 1], [2, 3, 4], [5, 6, 7], [8], [9], [10, 11], [12])   # S1, S2_RGB, Red_Edge, NIR_10m, NIR_20m, SWIR, NDVI
HALF_MONTH = 15 * DAY


# ------------------------------------------------------------------------------------------------------------------- time
def periods(w):
    """30-day steps from the window start; the last one ends at E (truncated)."""
    s = np.arange(int(w.start_ms), int(w.end_ms), PERIOD_MS, dtype=np.int64)
    return s, np.minimum(s + PERIOD_MS, int(w.end_ms))


def civil(ms):
    """(day 1-31, month 0-11, year) of UTC instants."""
    d = np.asarray(ms, dtype='datetime64[ms]').astype('datetime64[D]')
    m, y = d.astype('datetime64[M]'), d.astype('datetime64[Y]')
    return ((d - m.astype('datetime64[D]')).astype(np.int64) + 1, (m - y.astype('datetime64[M]')).astype(np.int64),
            y.astype(np.int64) + 1970)


def day_of_year(ms):
    d = np.asarray(ms, dtype='datetime64[ms]').astype('datetime64[D]')
    return (d - d.astype('datetime64[Y]').astype('datetime64[D]')).astype(np.int64) + 1


def orbit_pass(t_ms, lon):
    """S1 is dawn-dusk sun-synchronous: descending near 06 h, ascending near 18 h local solar time."""
    lst = ((np.asarray(t_ms, np.int64) % DAY) / HOUR + float(lon) / 15.0) % 24.0
    return np.where(lst >= 12.0, 'A', 'D')


# ------------------------------------------------------------------------------------------------------------------- field
def _readonly(x):
    """memoised reads are shared by every FM view of a chunk: made read-only so a view cannot alter another view's input"""
    if isinstance(x, np.ndarray):
        x.setflags(write=False)
    elif isinstance(x, dict):
        for v in x.values():
            _readonly(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _readonly(v)
    return x


class Field:
    """One field and window: shared per-field quantities (cached), pixel chunks read on demand. Reads of one pixel chunk are
    memoised, so building the views of several FMs for the same chunk reads S2 L2A, S1, L1C + CS+, DW and the background
    products once. Only the current chunk is kept; the per-field pre-passes use the raw readers."""

    def __init__(self, D, fid, w):
        self.D, self.fid, self.w = D, int(fid), w
        p = D.pixels(fid)
        self.lat, self.lon = p['lat'].astype(np.float64), p['lon'].astype(np.float64)
        self.n = len(self.lat)
        self.lon0 = float(np.median(self.lon))
        self.ps, self.pe = periods(w)
        self._cache = {}
        self._chunk, self._cc = None, {}

    def _memo(self, key, pos, fn):
        pos = np.asarray(pos)
        k = (len(pos), hash(pos.tobytes()))
        if self._chunk != k:
            self._chunk, self._cc = k, {}
        if key not in self._cc:
            self._cc[key] = _readonly(fn())
        return self._cc[key]

    def chunks(self, size=8192):
        for lo in range(0, self.n, size):
            yield np.arange(lo, min(self.n, lo + size))

    def _image(self, product, pos, window=True):
        w = self.w if window else None
        lat = LATENCY_H.get(product) if window else None
        return self.D.image(product, self.fid, w, latency_hours=lat, positions=pos)

    def image(self, product, pos, window=True):
        return self._memo(('image', product, window), pos, lambda: self._image(product, pos, window))

    # ---- S2 L2A (formal series), harmonised DN
    def l2a(self, pos):
        return self._memo(('l2a',), pos, lambda: self._l2a(pos))

    def _l2a(self, pos):
        s = self.D.s2(self.fid, self.w, positions=pos)
        v = s['values'][..., :12] + s['boa_add_offset'][:, None, None].astype(np.float64)
        ok = s['found'][..., :12] & s['valid'][..., :12] & np.isfinite(v) & (v >= 0)
        scl = np.where(s['found'][..., 12] & s['valid'][..., 12], s['values'][..., 12], -1.0)
        return s, v, ok, scl

    def l2a_scene_order(self, size=8192):
        """scene index order for OlmoEarth: field clear fraction (SCL 4/5, 12 bands) descending, then acquisition time"""
        if 'order' not in self._cache:
            num, t = 0, None
            for pos in self.chunks(size):
                s, v, ok, scl = self._l2a(pos)
                num = num + (np.isin(scl, SCL_CLEAR) & ok.all(-1)).sum(1)
                t = s["acquisition_ms"]
            self._cache['order'] = (np.lexsort((t, -num / max(self.n, 1))), num / max(self.n, 1))
        return self._cache['order']

    # ---- S1
    def s1_rtc(self, pos):
        return self._memo(('s1_rtc',), pos, lambda: self.D.s1_rtc(self.fid, self.w, latency_hours=48, positions=pos))

    def s1(self, pos):
        return self._memo(('s1',), pos, lambda: self._s1(pos))

    def _s1(self, pos):
        r = self._image('s1_grd', pos)
        b = r['bands']
        v = r['values'][..., [b.index('VV'), b.index('VH')]]
        ang = r['values'][..., b.index('angle')]
        g = r['found'] & r['valid']
        ok = g[..., b.index('VV')] & g[..., b.index('VH')] & np.isfinite(v).all(-1)
        okang = ok & g[..., b.index('angle')] & np.isfinite(ang) & (ang > 0) & (ang < 90)
        return dict(t=r['acquisition_ms'], keys=r['keys'], db=v, ok=ok, angle=ang, ok_angle=okang,
                    orbit=orbit_pass(r['acquisition_ms'], self.lon0))

    def s1_pass(self):
        """one orbit pass per field (official: one pass): the pass with more legal acquisitions, ties descending"""
        if 's1_pass' not in self._cache:
            r = self._image('s1_grd', np.zeros(1, np.int64))                 # image list only (one pixel read)
            o = orbit_pass(r['acquisition_ms'], self.lon0)
            self._cache['s1_pass'] = 'A' if (o == 'A').sum() > (o == 'D').sum() else 'D'
        return self._cache['s1_pass']

    def s1_steps(self, pos):
        """per 30-day step: median dB of the chosen pass's images nearest the step midpoint (official _get_closest_dates)"""
        return self._memo(('s1_steps',), pos, lambda: self._s1_steps(pos))

    def _s1_steps(self, pos):
        s = self.s1(pos)
        sel = s['orbit'] == self.s1_pass()
        T, n = len(self.ps), len(pos)
        out, have, used = np.full((n, T, 2), np.nan), np.zeros((n, T), bool), []
        x = np.where(s['ok'][..., None], s['db'], np.nan)
        for j, (a, b) in enumerate(zip(self.ps, self.pe)):
            d = np.abs(s['t'] - (a + b) // 2).astype(np.float64)
            d[~sel] = np.inf
            if not np.isfinite(d).any():
                used.append([])
                continue
            keep = np.flatnonzero(d <= max(HALF_MONTH, d.min()))
            used.append([s['keys'][i] for i in keep])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                out[:, j] = np.nanmedian(x[keep], axis=0)
            have[:, j] = np.isfinite(out[:, j]).all(-1)
        return np.where(have[..., None], out, 0.0), have, used

    # ---- S2 L1C quality mosaic (Galileo, Presto)
    def l1c_steps(self, pos, bands):
        return self._memo(('l1c_steps', tuple(bands)), pos, lambda: self._l1c_steps(pos, bands))

    def _l1c_base(self, pos):
        return self._memo(('l1c_base',), pos, lambda: self._l1c_score(pos))

    def _l1c_score(self, pos):
        r = self.image('s2_l1c', pos)
        c = self.image('cloud_score_plus', pos)
        rb = r['bands']
        need = [rb.index(b) for b in GALILEO_S2]
        g = r['found'] & r['valid']
        ok = g[..., need].all(-1) & np.isfinite(r['values'][..., need]).all(-1) & (r['values'][..., need] >= 0).all(-1)
        ck = {k: i for i, k in enumerate(c['keys'])}
        j_cdf = c['bands'].index('cs_cdf')
        score = np.full(ok.shape, -1.0)                                   # an acquisition without CS+ is used only when nothing else is
        for i, k in enumerate(r['keys']):
            jj = ck.get(k)
            if jj is not None:
                cg = c['found'][jj, :, j_cdf] & c['valid'][jj, :, j_cdf] & np.isfinite(c['values'][jj, :, j_cdf])
                score[i] = np.where(cg, c['values'][jj, :, j_cdf], -1.0)
        score = np.where(ok, score, -np.inf)
        return r, g, score, int(sum(k in ck for k in r['keys']))

    def _l1c_steps(self, pos, bands):
        r, g, score, matched = self._l1c_base(pos)
        rb = r['bands']
        bi = [rb.index(b) if b in rb else -1 for b in bands]
        T, n = len(self.ps), len(pos)
        out, have, src = np.zeros((n, T, len(bands))), np.zeros((n, T), bool), np.full((n, T), -1, np.int64)
        t = r['acquisition_ms']
        for j, (a, b) in enumerate(zip(self.ps, self.pe)):
            cand = np.flatnonzero((t >= a) & (t < b))
            if not len(cand):
                continue
            cand = cand[np.argsort(t[cand], kind='stable')]
            sc = score[cand]
            best = sc.argmax(0)                                           # first (earliest) of equal scores
            h = np.isfinite(sc[best, np.arange(n)])
            pick = cand[best]
            for q, k in enumerate(bi):
                if k >= 0:
                    vv = r['values'][pick, np.arange(n), k]
                    out[:, j, q] = np.where(h & g[pick, np.arange(n), k], vv, 0.0)
            have[:, j], src[:, j] = h, np.where(h, pick, -1)
        return out, have, dict(keys=r['keys'], source_index=src, cs_plus_matched=matched, images=len(r['keys']))

    # ---- ERA5-Land, field level (corner-weighted, every weighted cell valid)
    def era5_steps(self):
        if 'era5' in self._cache:
            return self._cache['era5']
        wx = self.D.weather('era5_land', self.fid, self.w)
        hours = np.unique(wx['time_ms'])
        series = {}
        for band in ('temperature_2m', 'total_precipitation_hourly'):
            j = wx['bands'].index(band)
            acc, good = np.zeros(len(hours)), np.ones(len(hours), bool)
            off, wt = wx['cell_offsets'], wx['weights']
            for k in np.flatnonzero(wt > 0):
                seg = slice(off[k], off[k + 1])
                x = np.full(len(hours), np.nan)
                x[np.searchsorted(hours, wx['time_ms'][seg])] = np.where(wx['valid'][seg, j], wx['values'][seg, j], np.nan)
                acc += wt[k] * np.nan_to_num(x)
                good &= np.isfinite(x)
            series[band] = np.where(good, acc, np.nan)
        T = len(self.ps)
        temp, rate, nh = np.full(T, np.nan), np.full(T, np.nan), np.zeros((T, 2), np.int64)
        for j, (a, b) in enumerate(zip(self.ps, self.pe)):
            ti = (hours >= a) & (hours < b) & np.isfinite(series['temperature_2m'])
            pi = (hours - HOUR >= a) & (hours <= b) & np.isfinite(series['total_precipitation_hourly'])
            nh[j] = ti.sum(), pi.sum()
            if ti.sum() >= 24:
                temp[j] = series['temperature_2m'][ti].mean()
            if pi.sum() >= 24:
                rate[j] = series['total_precipitation_hourly'][pi].sum() * 24.0 / pi.sum()   # m per day
        self._cache['era5'] = (temp, rate, nh)
        return self._cache['era5']

    # ---- monthly / annual background products (release not applied; `relaxed`)
    def monthly_steps(self, product, pos, bands):
        r = self.image(product, pos, window=False)
        legal = np.flatnonzero(r['interval_end_ms'] <= int(self.w.end_ms))
        T, n = len(self.ps), len(pos)
        out, have, used = np.zeros((n, T, len(bands))), np.zeros((n, T), bool), []
        if not all(b in r['bands'] for b in bands):
            return out, have, ['bands_not_in_dataset'] * T
        bi = [r['bands'].index(b) for b in bands]
        mid = (r['interval_start_ms'] + r['interval_end_ms']) // 2
        for j, (a, b) in enumerate(zip(self.ps, self.pe)):
            if not len(legal):
                used.append(None)
                continue
            i = legal[np.argmin(np.abs(mid[legal] - (a + b) // 2))]
            g = (r['found'][i][:, bi] & r['valid'][i][:, bi]).all(-1) & np.isfinite(r['values'][i][:, bi]).all(-1)
            out[:, j] = np.where(g[:, None], r['values'][i][:, bi], 0.0)
            have[:, j] = g
            used.append(r['keys'][i])
        return out, have, used

    def annual(self, product, pos, band):
        """nearest annual image to the window end (start <= E preferred); missing -> 0 as the official unmask(0)"""
        r = self.image(product, pos, window=False)
        if not len(r['keys']):
            return np.zeros(len(pos)), None
        s = r['interval_start_ms']
        cand = np.flatnonzero(s <= int(self.w.end_ms))
        i = cand[np.argmax(s[cand])] if len(cand) else int(np.argmin(s))
        k = r['bands'].index(band)
        g = r['found'][i][:, k] & r['valid'][i][:, k] & np.isfinite(r['values'][i][:, k])
        return np.where(g, r['values'][i][:, k], 0.0), r['keys'][i]

    def worldcereal(self, pos):
        return self._memo(('wc',), pos, lambda: self._worldcereal(pos))

    def _worldcereal(self, pos):
        """five products' classification at each pixel: official filter(product) -> mosaic() -> unmask(0). Keys are
        <AEZ>_TC-<season>_<PRODUCT>_<start>_<end>; mosaic draws later images over earlier ones, taken here in asset-ID order."""
        r = self._image('worldcereal_2021', pos, window=False)
        out = np.zeros((len(pos), len(WC_PRODUCTS)))
        k = r['bands'].index('classification')
        prod_of = [key.split('_')[2].lower() for key in r['keys']]
        for q, prod in enumerate(WC_PRODUCTS):
            for i in sorted((i for i, p in enumerate(prod_of) if p == prod), key=lambda i: r['keys'][i]):
                g = r['found'][i][:, k] & r['valid'][i][:, k] & np.isfinite(r['values'][i][:, k])
                out[:, q] = np.where(g, r['values'][i][:, k], out[:, q])
        return out

    def dw(self, pos):
        return self._memo(('dw',), pos, lambda: self._dw(pos))

    def _dw(self, pos):
        r = self._image('dynamic_world', pos)
        b = r['bands']
        ci = [b.index(c) for c in DW_CLASSES]
        g = (r['found'][..., ci] & r['valid'][..., ci]).all(-1) & np.isfinite(r['values'][..., ci]).all(-1)
        return r, np.where(g[..., None], r['values'][..., ci], np.nan), g

    def dw_static(self, size=8192):
        """field mean of the per-pixel window-mean DW probabilities and of WorldCereal (official static = tile mean)"""
        if 'dw_static' not in self._cache:
            sdw, ndw, swc = np.zeros(len(DW_CLASSES)), 0, np.zeros(len(WC_PRODUCTS))
            for pos in self.chunks(size):
                _, p, g = self._dw(pos)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    m = np.nanmean(p, axis=0)
                ok = g.any(0)
                sdw += m[ok].sum(0)
                ndw += int(ok.sum())
                swc += self._worldcereal(pos).sum(0)
            self._cache['dw_static'] = (sdw / ndw if ndw else None, swc / self.n)
        return self._cache['dw_static']

    def tc_steps(self, n):
        """TerraClimate def soil aet of the month nearest each step midpoint among months ending by E (release not applied;
        `relaxed`). The official export feeds GEE's stored integers (scale 0.1 not applied; normalization.json means 657 / 692 /
        563); the dataset holds the scaled physical values, so value x 10 is used, accepted only when it is an integer (exact)."""
        T = len(self.ps)
        meta = self.D.manifest['meta']['products'].get('terraclimate', {})
        bands = meta.get('scalars', {}).get('bands', [])
        out, have = np.zeros((n, T, 3)), np.zeros((n, T), bool)
        if not all(b in bands for b in TC_BANDS):
            return out, have, 'def_soil_aet_not_all_in_dataset'
        p = self.D.product('terraclimate', self.fid)
        legal = np.flatnonzero(p['time_end_ms'] <= int(self.w.end_ms))
        mid = (p['time_start_ms'] + p['time_end_ms']) // 2
        bi = [p['bands'].index(b) for b in TC_BANDS]
        used = []
        for j, (a, b) in enumerate(zip(self.ps, self.pe)):
            if not len(legal):
                used.append(None)
                continue
            i = legal[np.argmin(np.abs(mid[legal] - (a + b) // 2))]
            x = np.asarray(p['values'][i], np.float64)[bi] * 10.0
            raw = np.round(x)
            ok = (bool((np.asarray(p['valid'][i])[bi] & np.asarray(p['quality_valid'][i])[bi]).all()) and np.isfinite(x).all()
                  and bool((np.abs(x - raw) < 1e-6).all()))
            if ok:
                out[:, j], have[:, j] = raw, True
            used.append(p['image_id'][i])
        return out, have, used

    def srtm(self, pos):
        if 'static' not in self._cache:
            st = self.D.static(self.fid)
            self._cache['static'] = (st['terrain_elevation_m'].astype(np.float64), st['terrain_slope_rise'].astype(np.float64))
        e, s = (a[pos] for a in self._cache['static'])
        ok = np.isfinite(e) & np.isfinite(s)
        return np.where(ok, e, 0.0), np.where(ok, np.degrees(np.arctan(np.where(ok, s, 0.0))), 0.0), ok


# ------------------------------------------------------------------------------------------------------------------- views
def tessera_rtc_scale(power, valid):
    """Exact vendored amplitude_to_db operations, including float32 and 32767 cap.

    MPC calls its values gamma0 power; this reproduces TESSERA's preprocessing
    (20*log10), not the physical power-to-dB formula (10*log10).
    """
    power = np.asarray(power, np.float32)
    good = np.asarray(valid, bool) & np.isfinite(power) & (power > 0)
    out = np.zeros(power.shape, np.int16)
    out[good] = np.clip((20.0 * np.log10(power[good]) + 50.0) * 200.0, 0, 32767).astype(np.int16)
    return out.astype(np.float32)


def tessera(F, pos):
    s, v, ok, scl = F.l2a(pos)
    ix = [L2A.index(b) for b in TESSERA_S2]
    clear = (scl >= 0) & ~np.isin(scl, SCL_INVALID_TESSERA) & ok[..., ix].all(-1)            # [images, pixels]
    s2 = np.where(clear[..., None], v[..., ix], 0.0).transpose(1, 0, 2).astype(np.float32)
    r = F.s1_rtc(pos)
    val = tessera_rtc_scale(r['values'], r['found'] & r['valid']).transpose(1, 0, 2)
    doy1 = day_of_year(r['acquisition_ms'])
    out = dict(model='tessera', s2_bands=s2, s2_doys=day_of_year(s['acquisition_ms']), s2_masks=clear.T,
               has_observation=clear.any(0), positions=pos, s2_image_ids=s['image_ids'], relaxed=[])
    for o, name in (('A', 'asc'), ('D', 'desc')):
        k = np.flatnonzero(r['orbit'] == o)
        out[f's1_{name}_bands'], out[f's1_{name}_doys'] = val[:, k], doy1[k]
    out['s1_valid'] = np.any(val != 0, axis=-1)  # official encoder treats all-zero scaled frames as absent
    out['s1_image_ids'] = r['keys']
    out['s1_unavailable'] = r['unavailable']
    out['s1_source'] = 'MPC RTC gamma0; validated native windows'
    return out


def galileo(F, pos):
    n, T = len(pos), len(F.ps)
    stx, stm = np.zeros((n, T, 13)), np.ones((n, T, 7), np.uint8)
    s1, h1, s1_used = F.s1_steps(pos)
    stx[..., 0:2], stm[..., 0] = s1, ~h1
    s2, h2, l1c_info = F.l1c_steps(pos, GALILEO_S2)
    stx[..., 2:12] = s2
    b8, b4 = s2[..., GALILEO_S2.index('B8')], s2[..., GALILEO_S2.index('B4')]
    den = b8 + b4
    stx[..., 12] = np.where(h2 & (den > 0), (b8 - b4) / np.where(den > 0, den, 1.0), 0.0)
    stm[..., 1:] = ~h2[..., None]
    tx, tm = np.zeros((n, T, 6)), np.ones((n, T, 3), np.uint8)
    temp, rate, _ = F.era5_steps()
    he = np.isfinite(temp) & np.isfinite(rate)
    tx[:, he, 0], tx[:, he, 1], tm[:, he, 0] = temp[he], rate[he] * 30.0, 0          # monthly-sum scale of a 30-day step
    tc, htc, tc_used = F.tc_steps(n)
    tx[..., 2:5], tm[..., 1] = tc, ~htc
    vi, hvi, vi_used = F.monthly_steps('viirs_dnb_monthly', pos, ('avg_rad',))
    tx[..., 5], tm[..., 2] = vi[..., 0], ~hvi
    spx, spm = np.zeros((n, 16)), np.ones((n, 3), np.uint8)
    e, sl, hs = F.srtm(pos)
    spx[:, 0], spx[:, 1], spm[:, 0] = e, sl, ~hs
    _, p, g = F.dw(pos)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        m = np.nanmean(p, axis=0)
    hdw = g.any(0)
    spx[:, 2:11], spm[:, 1] = np.where(hdw[:, None], m, 0.0), ~hdw
    spx[:, 11:16], spm[:, 2] = F.worldcereal(pos), 0
    sx, sm = np.zeros((n, 18)), np.ones((n, 4), np.uint8)
    ls, ls_key = F.annual('landscan_global', pos, 'b1')
    sx[:, 0], sm[:, 0] = ls, 0 if ls_key is not None else 1
    la, lo = np.radians(F.lat[pos]), np.radians(F.lon[pos])
    sx[:, 1:4], sm[:, 1] = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1), 0
    dws, wcs = F.dw_static()
    if dws is not None:
        sx[:, 4:13], sm[:, 2] = dws, 0
    sx[:, 13:18], sm[:, 3] = wcs, 0
    months = civil(F.ps)[1]
    return dict(model='galileo', s_t_x=stx, s_t_m=stm, t_x=tx, t_m=tm, sp_x=spx, sp_m=spm, st_x=sx, st_m=sm, months=months,
                has_observation=(h1 | h2).any(1), positions=pos, s2_steps=h2.sum(1), s1_steps=h1.sum(1),
                provenance=dict(s1_pass=F.s1_pass(), s1_images=s1_used, l1c=l1c_info, viirs=vi_used, tc=tc_used, landscan=ls_key),
                relaxed=['viirs_release', 'terraclimate_release', 'landscan_release', 'worldcereal_2021_any_season'])


def presto(F, pipeline, pos):
    """raw x in the official BANDS order -> official S1_S2_ERA5_SRTM.normalize; group mask; DW label mode; lat / lon"""
    n, T = len(pos), len(F.ps)
    B = list(pipeline.BANDS)
    raw = np.zeros((n, T, len(B)), np.float32)
    s1, h1, s1_used = F.s1_steps(pos)
    raw[..., B.index('VV')], raw[..., B.index('VH')] = s1[..., 0], s1[..., 1]
    s2, h2, l1c_info = F.l1c_steps(pos, PRESTO_S2)
    for q, b in enumerate(PRESTO_S2):
        raw[..., B.index(b)] = s2[..., q]
    temp, rate, _ = F.era5_steps()
    he = np.isfinite(temp) & np.isfinite(rate)
    raw[:, he, B.index('temperature_2m')], raw[:, he, B.index('total_precipitation')] = temp[he], rate[he]
    e, sl, hs = F.srtm(pos)
    raw[..., B.index('elevation')], raw[..., B.index('slope')] = e[:, None], sl[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        x = pipeline.S1_S2_ERA5_SRTM.normalize(raw)
    NB = list(pipeline.NORMED_BANDS)
    mask = np.zeros(x.shape, np.float32)
    groups = pipeline.BANDS_GROUPS_IDX
    for name, idx in groups.items():
        miss = (~h1 if name == 'S1' else ~h2 if name.startswith('S2') or name == 'NDVI' else
                np.broadcast_to(~he[None, :], (n, T)) if name == 'ERA5' else np.broadcast_to(~hs[:, None], (n, T)) if name == 'SRTM' else None)
        if miss is None:
            raise ValueError(f'unknown Presto band group {name}')
        mask[..., idx] = miss[..., None]
    assert set(i for idx in groups.values() for i in idx) == set(range(len(NB))), 'Presto groups must cover every normed band'
    x = np.where(mask == 0, x, 0.0).astype(np.float32)
    r, p, g = F.dw(pos)
    k = r['bands'].index('label')                                             # official: mode of the DW 'label' band
    gl = r['found'][..., k] & r['valid'][..., k] & np.isfinite(r['values'][..., k])
    lab = np.where(gl, np.nan_to_num(r['values'][..., k], nan=-1.0), -1).astype(np.int64)
    lab = np.where((lab >= 0) & (lab <= 8), lab, -1)
    t = r['acquisition_ms']
    dw = np.full((n, T), 9, np.int64)
    for j, (a, b) in enumerate(zip(F.ps, F.pe)):
        k = np.flatnonzero((t >= a) & (t < b))
        if not len(k):                                                        # official DynamicWorldMonthly: nearest within 15 days
            d = np.abs(t - (a + b) // 2)
            k = np.flatnonzero(d <= HALF_MONTH)
            k = k[np.argsort(d[k], kind='stable')][:1] if len(k) else k
        if len(k):
            cnt = np.zeros((n, 9), np.int64)
            for i in k:
                ok = lab[i] >= 0
                cnt[np.flatnonzero(ok), lab[i][ok]] += 1
            have = cnt.sum(1) > 0
            dw[have, j] = cnt[have].argmax(1)
    latlon = np.stack([F.lat[pos], F.lon[pos]], -1).astype(np.float32)
    return dict(model='presto', x=x, mask=mask, dynamic_world=dw, latlons=latlon, month=int(civil(F.ps[:1])[1][0]),
                has_observation=(h1 | h2).any(1), positions=pos, s2_steps=h2.sum(1), s1_steps=h1.sum(1),
                provenance=dict(s1_pass=F.s1_pass(), s1_images=s1_used, l1c=l1c_info), relaxed=[])


def olmoearth(F, pos):
    """sentinel2_l2a and sentinel1 per 30-day period; values before the official COMPUTED normalisation; -99999 where missing"""
    n, T = len(pos), len(F.ps)
    s, v, ok, scl = F.l2a(pos)
    ix = [L2A.index(b) for b in OLMO_S2]
    clear = np.isin(scl, SCL_CLEAR) & ok[..., ix].all(-1)
    order, frac = F.l2a_scene_order()
    t = s['acquisition_ms']
    s2 = np.full((n, T, 12), OLMO_MISSING)
    tok2 = np.zeros((n, T), bool)
    for j, (a, b) in enumerate(zip(F.ps, F.pe)):
        cand = order[(t[order] >= a) & (t[order] < b)]
        if not len(cand):
            continue
        good = clear[cand]
        first = good.argmax(0)
        h = good[first, np.arange(n)]
        pick = cand[first]
        s2[h, j] = v[pick[h], np.flatnonzero(h)][:, ix]
        tok2[:, j] = h
    r = F.s1(pos)
    s1 = np.full((n, T, 2), OLMO_MISSING)
    tok1 = np.zeros((n, T), bool)
    for j, (a, b) in enumerate(zip(F.ps, F.pe)):
        cand = np.flatnonzero((r['t'] >= a) & (r['t'] < b))
        if not len(cand):
            continue
        cand = cand[np.argsort(r['t'][cand], kind='stable')]
        good = r['ok'][cand]
        first = good.argmax(0)
        h = good[first, np.arange(n)]
        pick = cand[first]
        s1[h, j] = r['db'][pick[h], np.flatnonzero(h)]
        tok1[:, j] = h
    d, m, y = civil(F.ps)
    return dict(model='olmoearth', sentinel2_l2a=s2[:, None, None].astype(np.float32), sentinel2_l2a_token=tok2[:, None, None, :, None],
                sentinel1=s1[:, None, None].astype(np.float32), sentinel1_token=tok1[:, None, None, :, None],
                timestamps=np.stack([d, m, y], -1).astype(np.int64), has_observation=(tok2 | tok1).any(1), positions=pos,
                s2_periods=tok2.sum(1), s1_periods=tok1.sum(1), scene_clear_fraction=frac, relaxed=[])
