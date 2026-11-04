"""Pixel samples of the Base models: Sentinel-2 time series, daily weather, soil, terrain and labels.

A case is one field-season observed up to a window end. Its inputs:
  s2:      [pixels, steps, 12] harmonised reflectance (float32), scl: [pixels, steps] scene classes, ok: [pixels, steps] valid
           observation (all 12 bands found and valid), day: [steps] UTC day numbers;
  weather: [days, 4] daily mean, max, min air temperature (K) and precipitation (m), weather_day: [days] UTC day numbers;
  input_start_day: UTC day number of the field-season input start (CASES.input_start_utc_ms), the same for every decision
           time of the field-season;
  pix0:    index of the field's first pixel in the static arrays (soil [P, 24], dem [P, 5], target [P], gt_valid [P]).
UTC day numbers count days since 1970-01-01. The Transformer batch gives the Sentinel-2 and weather dates as days since
input_start_day; the 3D-LSTM date features use the UTC day numbers. Invalid steps get date DATE_PAD and are masked in the
temporal encoders; masked values are -1 after standardisation.
"""
import numpy as np
import torch

DATE_PAD, SCL_PAD, NB = -1000, 12, 12


class PixelData:
    """All samples of a set of cases on one device: flat steps plus per-sample (start, length, case, pixel)."""

    def __init__(self, cases, soil, dem, target, gt_valid, samples='gt', device='cpu'):
        vals, scl, ok, day, s_start, s_len, s_case, s_pix, w_list = [], [], [], [], [], [], [], [], []
        n = 0
        self.case_ids = [c['case_id'] for c in cases]
        self.start_day = torch.tensor([int(c['input_start_day']) for c in cases], dtype=torch.int32, device=device)
        for ci, c in enumerate(cases):
            raw, sc, k, d = c['s2'], c['scl'], c['ok'], c['day']
            n_pix = raw.shape[0]
            pix = np.arange(n_pix)
            keep = gt_valid[c['pix0'] + pix] if samples == 'gt' else np.ones(n_pix, bool)
            pix = pix[keep]
            L = raw.shape[1]
            vals.append(raw[keep].astype(np.float32).reshape(-1, NB))
            scl.append(sc[keep].reshape(-1))
            ok.append(k[keep].reshape(-1))
            day.append(np.broadcast_to(np.asarray(d)[None, :], (len(pix), L)).reshape(-1))
            s_start.append(n + np.arange(len(pix)) * L)
            s_len.append(np.full(len(pix), L))
            s_case.append(np.full(len(pix), ci))
            s_pix.append(c['pix0'] + pix)
            n += len(pix) * L
            w_list.append((c['weather'], c['weather_day']))
        cat = lambda x, dt: torch.from_numpy(np.concatenate(x).astype(dt)) if x else torch.zeros(0)  # noqa: E731
        self.vals = cat(vals, np.float32).reshape(-1, NB).to(device)
        self.scl = cat(scl, np.uint8).to(device)
        self.ok = cat(ok, bool).to(device)
        self.day = cat(day, np.int32).to(device)
        self.s_start, self.s_len = cat(s_start, np.int64).to(device), cat(s_len, np.int64).to(device)
        self.s_case, self.s_pix = cat(s_case, np.int64).to(device), cat(s_pix, np.int64).to(device)
        W = max([len(d) for _, d in w_list] + [1])
        wv = np.full((len(w_list), W, 4), np.nan, np.float32)
        wd = np.full((len(w_list), W), DATE_PAD, np.float32)
        for k, (x, d) in enumerate(w_list):
            wv[k, :len(d)], wd[k, :len(d)] = x, d
        wd[~np.isfinite(wv).all(-1)] = DATE_PAD
        self.wv, self.wd = torch.from_numpy(wv).to(device), torch.from_numpy(wd).to(device)
        self.soil = torch.from_numpy(soil).to(device)
        self.dem = torch.from_numpy(dem).to(device)
        self.target = torch.from_numpy(target).to(device)
        self.gt_valid = torch.from_numpy(gt_valid).to(device)
        self.n = len(self.s_len)


def _nanstd(x):
    m = torch.nanmean(x, 0, keepdim=True)
    return torch.sqrt(torch.nanmean((x - m) ** 2, 0))


def fit_stats(p, sel=None, chunk=1 << 24):
    """Mean and standard deviation per channel over the fitting samples sel (default all).

    S2 over their valid steps, weather over the valid days of their cases (each case once), soil, terrain and target over
    the samples.
    """
    dev = p.vals.device
    sel = torch.ones(p.n, dtype=torch.bool, device=dev) if sel is None else sel
    m = torch.repeat_interleave(sel, p.s_len) & p.ok if len(p.ok) else p.ok
    s = torch.zeros(NB, dtype=torch.float64, device=dev)
    ss, n = torch.zeros_like(s), 0
    for a in range(0, len(m), chunk):
        x = p.vals[a:a + chunk][m[a:a + chunk]].double()
        s += x.sum(0)
        ss += (x * x).sum(0)
        n += len(x)
    mu = s / max(n, 1)
    sd = torch.sqrt(torch.clamp(ss / max(n, 1) - mu * mu, min=0))

    def fix(mu, sd):
        mu, sd = torch.nan_to_num(mu.float(), nan=0.0), torch.nan_to_num(sd.float(), nan=1.0)
        return mu, torch.where(sd > 1e-6, sd, torch.ones_like(sd))
    cases = torch.unique(p.s_case[sel])
    wv, wd = p.wv[cases].reshape(-1, 4), p.wd[cases].reshape(-1)
    w = wv[(wd != DATE_PAD) & torch.isfinite(wv).all(-1)]
    pix = p.s_pix[sel]
    soil_x, dem_x, t = p.soil[pix], p.dem[pix], p.target[pix]
    return dict(s2=fix(mu, sd), wx=fix(w.mean(0), w.std(0, unbiased=False)), soil=fix(torch.nanmean(soil_x, 0), _nanstd(soil_x)),
                dem=fix(torch.nanmean(dem_x, 0), _nanstd(dem_x)), y=(t.mean(), t.std(unbiased=False)))


def batch(p, idx, st):
    """Transformer batch for samples idx, padded to the longest sample.

    Sentinel-2 and weather dates are days since the input start of the sample's field-season (the same origin at every
    decision time); invalid steps and days are DATE_PAD.
    """
    L = max(int(p.s_len[idx].max().item()) if len(idx) else 0, 1)
    ar = torch.arange(L, device=p.vals.device)
    inside = ar[None, :] < p.s_len[idx][:, None]
    g = (p.s_start[idx][:, None] + ar[None, :]).clamp(max=max(len(p.ok) - 1, 0))
    ok = inside & p.ok[g] if len(p.ok) else inside & False
    x = (p.vals[g] - st['s2'][0]) / st['s2'][1] if len(p.ok) else torch.zeros(len(idx), L, NB, device=ar.device)
    x = torch.where(ok[..., None], x, torch.full_like(x, -1.0))
    scl = torch.where(inside, p.scl[g].long() if len(p.ok) else torch.full_like(ar[None].expand(len(idx), L), SCL_PAD), SCL_PAD)
    scl = torch.where(ok, scl, SCL_PAD)
    oh = torch.nn.functional.one_hot(scl.clamp(0, SCL_PAD), SCL_PAD + 1).float()
    c = p.s_case[idx]
    start = p.start_day[c][:, None]
    day = (p.day[g] - start).float() if len(p.ok) else torch.zeros(len(idx), L, device=ar.device)
    dates = torch.where(ok, day, torch.full_like(day, DATE_PAD))
    wv = (p.wv[c] - st['wx'][0]) / st['wx'][1]
    wd = p.wd[c]
    wd = torch.where(wd != DATE_PAD, wd - start.float(), wd)
    wv = torch.where((wd != DATE_PAD)[..., None], torch.nan_to_num(wv, nan=-1.0), torch.full_like(wv, -1.0))
    soil = torch.nan_to_num((p.soil[p.s_pix[idx]] - st['soil'][0]) / st['soil'][1], nan=0.0)
    dem = torch.nan_to_num((p.dem[p.s_pix[idx]] - st['dem'][0]) / st['dem'][1], nan=0.0)
    y = (p.target[p.s_pix[idx]] - st['y'][0]) / st['y'][1]
    return {'x_s2': torch.cat([x, oh], -1), 'x_s2_dates': dates, 'x_weather': wv, 'x_weather_dates': wd,
            'x_soil': soil, 'x_dem': dem, 'target': y}


def bucket_batches(lengths, batch_size, rng, chunk=64):
    """Shuffled batches of similar length: permute, sort within chunks of batches by length, shuffle the batch order."""
    perm = rng.permutation(len(lengths))
    out = []
    step = batch_size * chunk
    for s in range(0, len(perm), step):
        blk = perm[s:s + step]
        blk = blk[np.argsort(lengths[blk], kind='stable')]
        out += [blk[i:i + batch_size] for i in range(0, len(blk), batch_size)]
    order = rng.permutation(len(out))
    return [out[k] for k in order]
