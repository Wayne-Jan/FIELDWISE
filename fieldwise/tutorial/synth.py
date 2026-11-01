"""Synthetic yields for the tutorial (rule frozen on 2026-11-01 before any fit; see the tutorial README).

Only the yields are synthetic; every input is a real public observation. Field-season level (z over all field-seasons):
    F_f = 6.0 + 0.8 z(mean peak NDVI) + 0.5 z(season rain) - 0.4 z(hot days) + a_year + b_field-season
Pixel level (z_f within the field-season):
    y_p = clip(F_f + 0.5 z_f(peak NDVI_p) + 0.4 z_f(e_p . v) + eps_p, 0.5, 15)  [t/ha]
peak NDVI: maximum valid Sentinel-2 NDVI over the season (harvest-time case window); rain: season precipitation (mm); hot
days: days with maximum air temperature above 30 C; e_p: TESSERA representation of the first decision window (tau1); v: a
fixed random unit vector. a ~ N(0, 0.5^2) per year, b ~ N(0, 0.4^2) per field-season, eps ~ N(0, 0.3^2) per pixel, drawn
in that order (after v) from numpy default_rng(20261101). Pixels without a finite peak NDVI or representation are not
labelled.
"""
import numpy as np
import pandas as pd

SEED = 20261101
B4, B8 = 3, 7                                   # positions in the 12 Sentinel-2 L2A bands of the Base inputs


def z(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / np.nanstd(x)


def generate(study):
    """(target [pixels], gt_valid [pixels], components DataFrame per pixel, field-season table)."""
    C = study.cases
    H = C[C.stage == 'harvest'].sort_values(['year', 'fid']).reset_index(drop=True)
    T1 = C[C.stage == 'tau1'].set_index(['year', 'fid'])
    peak = np.full(study.n_pix, np.nan)
    rain, hot = np.zeros(len(H)), np.zeros(len(H))
    for i, x in enumerate(study.case_inputs(H)):                                # one harvest case per field-season
        s2 = x['s2']
        ndvi = np.where(x['ok'], (s2[..., B8] - s2[..., B4]) / np.maximum(s2[..., B8] + s2[..., B4], 1), np.nan)
        with np.errstate(all='ignore'):
            peak[x['pix0'] + np.arange(len(s2))] = np.nanmax(np.where(np.isfinite(ndvi), ndvi, -np.inf), 1)
        w = x['weather']
        rain[i] = np.nansum(w[:, 3]) * 1000.0
        hot[i] = np.sum(w[:, 1] - 273.15 > 30.0)
    peak[~np.isfinite(peak)] = np.nan
    row, pix = study.pixel_index(H)                                             # pixel -> field-season row
    e = np.full((study.n_pix, study.emb.shape[1]), np.nan, np.float32)
    ix = study.emb_index.set_index('case_id')
    for j, r in enumerate(H.itertuples()):
        t1 = T1.loc[(r.year, r.fid)]
        a = int(ix.loc[t1.case_id, 'row_start'])
        e[r.gpix0:r.gpix0 + r.n_pix] = study.emb[a:a + r.n_pix]
    rng = np.random.default_rng(SEED)
    v = rng.normal(size=study.emb.shape[1])
    v /= np.linalg.norm(v)
    years = sorted(H.year.unique())
    a_y = dict(zip(years, rng.normal(0, 0.5, len(years))))
    b_f = rng.normal(0, 0.4, len(H))
    eps = rng.normal(0, 0.3, study.n_pix)
    mean_peak = pd.Series(peak[pix]).groupby(row).mean().reindex(range(len(H))).to_numpy()
    H = H.assign(mean_peak=mean_peak, rain_mm=rain, hot_days=hot, a_year=H.year.map(a_y).to_numpy(), b_fs=b_f)
    H['F_obs'] = 6.0 + 0.8 * z(H.mean_peak) + 0.5 * z(H.rain_mm) - 0.4 * z(H.hot_days)
    H['F'] = H.F_obs + H.a_year + H.b_fs
    proj = e[pix] @ v
    valid = np.isfinite(peak[pix]) & np.isfinite(proj)
    df = pd.DataFrame(dict(row=row, pix=pix, peak=peak[pix], proj=np.where(valid, proj, np.nan)))
    zf = lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else s * 0  # noqa: E731
    df['w_ndvi'] = 0.5 * df.groupby('row').peak.transform(zf)
    df['w_emb'] = 0.4 * df.groupby('row').proj.transform(zf)
    df['eps'] = eps[pix]
    df['F'] = H.F.to_numpy()[row]
    y = np.clip(df.F + df.w_ndvi + df.w_emb + df.eps, 0.5, 15.0).to_numpy()
    target = np.full(study.n_pix, np.nan, np.float32)
    gt = np.zeros(study.n_pix, bool)
    target[pix[valid]] = y[valid]
    gt[pix[valid]] = True
    return target, gt, df, H
