"""Several collected seasons as one study.

A season folder holds the outputs of one collection run (python -m fieldwise.collect): collect/ (fields, base_inputs,
g_inputs) and encode/ (tessera_pixels.npy, tessera_index.parquet, fm_means.npz); the retrieval descriptors are computed
here from the Base inputs (descriptors.py). A field-season
is (year, fid); case ids carry the year, so they are unique across seasons. Pixels of all seasons are concatenated in the
order of the seasons given (gpix0 = global index of a case's first pixel).

The season start and end are the dates of FIELDS.parquet (sow_day, harvest_day): sowing and harvest dates when known, the
image-derived proxies otherwise (date_source). Labels count as available LABEL_DELAY_DAYS after the end.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import weather
from .descriptors import season_descriptors
from ..schedule import DAY_MS, LABEL_DELAY_DAYS

NAMED = ('E1_total', 'E1_14d', 'E2_total', 'E2_14d', 'E4_cum_precip', 'E3_theta_last7', 'E3_theta_min', 'lst_day_14',
         'dT_day_14', 'dT_day_seas', 'dT_pos_30', 'lst_night_14')


class Season:
    """Base inputs of one collected season, read like inputs.Prepared."""

    def __init__(self, folder):
        d = self.folder = Path(folder)
        b = d / 'collect/base_inputs'
        self.cases = pd.read_parquet(b / 'CASES.parquet')
        with np.load(b / 'corn_static.npz') as z:
            self.soil, self.dem, self.fid, self.pix0 = z['soil'], z['dem'], z['fid'], z['pix0']
        with np.load(b / 'corn_grid.npz') as z:
            key = pd.DataFrame(dict(field=np.repeat(z['fid'] - 900_000, np.diff(z['pix0'])), row=z['row'], col=z['col']))
        px = pd.read_parquet(d / 'collect/fields/PIXELS.parquet')
        ll = key.merge(px[['field', 'row', 'col', 'lon', 'lat']], on=['field', 'row', 'col'], how='left', validate='one_to_one')
        self.lon, self.lat = ll.lon.to_numpy(), ll.lat.to_numpy()
        w = np.load(b / 'weather_daily.npz')
        self.w_pos = {int(f): k for k, f in enumerate(w['fid'])}
        self.w_start, self.w_day, self.w_val = w['w_start'], w['day'], w['values']
        self.img_start, self.img_day, self.img_off = w['img_start'], w['img_day'], w['img_offset']
        self.pack = pd.read_parquet(b / 's2_pack/PACK_INDEX.parquet').set_index('fid')
        self.V, self.S, self.K = (np.load(b / f's2_pack/corn_{n}.npy', mmap_mode='r') for n in ('vals', 'scl', 'ok'))
        self.fields = (pd.read_parquet(d / 'collect/fields/FIELDS.parquet').assign(fid=lambda f: 900_000 + f.field)
                       .set_index('fid'))

    def case_inputs(self, c, gpix0):
        """Base input dict of one case (as inputs.Prepared._case), with pix0 set to the global pixel index."""
        k = self.w_pos[int(c.fid)]
        s, e = self.img_start[k], self.img_start[k + 1]
        day, off = self.img_day[s:e], self.img_off[s:e]
        r = self.pack.loc[int(c.fid)]
        n_pix, L, o = int(r.n_pix), int(r.L), int(r.off)
        a, b = int(c.i0) - int(r.i0f), int(c.i1) - int(r.i0f)
        assert 0 <= a <= b <= L and n_pix == int(c.n_pix), f'{c.case_id}: case window outside the S2 pack'
        sl = slice(o, o + n_pix * L)
        raw = np.asarray(self.V[sl]).reshape(n_pix, L, -1)[:, a:b]
        ws = self.w_start[k]
        return dict(case_id=c.case_id, pix0=int(gpix0), s2=raw.astype(np.float32) + off[int(c.i0):int(c.i1)][None, :, None],
                    scl=np.asarray(self.S[sl]).reshape(n_pix, L)[:, a:b], ok=np.asarray(self.K[sl]).reshape(n_pix, L)[:, a:b],
                    day=day[int(c.i0):int(c.i1)].astype(np.int32), weather=self.w_val[ws + int(c.w0):ws + int(c.w1)],
                    weather_day=self.w_day[ws + int(c.w0):ws + int(c.w1)], input_start_day=int(c.input_start_utc_ms) // DAY_MS)


class Study:
    """All seasons: case table, static pixel arrays, environment, embeddings and retrieval descriptors."""

    def __init__(self, folders):
        self.seasons = [Season(f) for f in folders]
        offs = np.cumsum([0] + [len(s.soil) for s in self.seasons])
        parts, fs_parts, daily, fs_off = [], [], [], 0
        named, q, means, emb, index = [], [], {}, [], []
        row0 = 0
        for i, s in enumerate(self.seasons):
            c = s.cases.rename(columns={'decision_utc_ms': 'decision_ms', 'window_end_utc_ms': 'window_end_ms'}).copy()
            f = s.fields.loc[c.fid]
            c['season'], c['field'] = i, c.fid
            c['gpix0'] = c.pix0 + offs[i]
            c['start_day'], c['end_day'] = f.sow_day.to_numpy(), f.harvest_day.to_numpy()
            c['label_ms'] = ((c.end_day + LABEL_DELAY_DAYS) * DAY_MS).astype(float)
            g = s.folder / 'collect/g_inputs'
            gc = pd.read_parquet(g / 'cases.parquet').set_index('case_id').loc[c.case_id]
            c['fs_row'], c['n_days'] = gc.fs_row.to_numpy() + fs_off, gc.n_days.to_numpy()
            fs = pd.read_parquet(g / 'fs.parquet')
            fs_parts.append(fs.assign(season=i))
            daily.append(weather.daily_quantities(np.load(g / 'X.npy', mmap_mode='r'),
                                                  json.loads((g / 'PACK.json').read_text())['bands']))
            fs_off += len(fs)
            E = pd.read_parquet(g / 'E_TAIWAN.parquet').join(pd.read_parquet(g / 'LST_TAIWAN.parquet'))
            named.append(E.reindex(c.case_id)[list(NAMED)])
            e = s.folder / 'encode'
            z = np.load(e / 'fm_means.npz')
            means.update(zip(z['case_ids'], z['means']))
            self.fm_models = [str(m) for m in z['models']]
            ix = pd.read_parquet(e / 'tessera_index.parquet')
            emb.append(np.load(e / 'tessera_pixels.npy'))
            index.append(ix.assign(row_start=ix.row_start + row0))
            row0 += len(emb[-1])
            q.append(season_descriptors(s))
            parts.append(c)
        self.cases = pd.concat(parts, ignore_index=True)
        self.soil = np.concatenate([s.soil for s in self.seasons]).astype(np.float32)
        self.dem = np.concatenate([s.dem for s in self.seasons]).astype(np.float32)
        self.lon = np.concatenate([s.lon for s in self.seasons])
        self.lat = np.concatenate([s.lat for s in self.seasons])
        self.n_pix = len(self.soil)
        self.fs = pd.concat(fs_parts, ignore_index=True)
        L = max(d.shape[1] for d in daily)
        self.daily = np.concatenate([np.pad(d, ((0, 0), (0, L - d.shape[1]), (0, 0)), constant_values=np.nan) for d in daily])
        self.named = pd.concat(named)
        self.fm_means = means
        self.emb = np.concatenate(emb)
        self.emb_index = pd.concat(index, ignore_index=True)
        self.q = pd.concat(q, ignore_index=True).set_index('case_id')

    def case_inputs(self, cases):
        """Base input dicts of the given rows of the case table."""
        return [self.seasons[int(c.season)].case_inputs(c, c.gpix0) for c in cases.itertuples()]

    def pixel_index(self, cases):
        """(row, global pixel) of every service pixel of the given cases, rows indexing cases in their order."""
        n = cases.n_pix.to_numpy()
        row = np.repeat(np.arange(len(cases)), n)
        start = np.repeat(cases.gpix0.to_numpy() - np.r_[0, np.cumsum(n)[:-1]], n)
        return row, start + np.arange(n.sum())
