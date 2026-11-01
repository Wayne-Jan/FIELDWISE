"""Base yield models: a multimodal Transformer and a 3D-LSTM over the same four input modalities.

Transformer (after Najjar et al., 2025): Sentinel-2 (12 bands + 13 scene-class channels) and daily weather (4 channels) each
pass a temporal Transformer (sinusoidal dates, learned regression token, four one-head layers); soil (24) and terrain (5)
pass small MLPs; the four 32-dimensional outputs feed a linear regression.
3D-LSTM: per valid Sentinel-2 step the 5 x 5 same-field neighbourhood of the 12 bands (300 values) is projected to 64
features, joined with the scene class and date features and passed to a two-layer LSTM; weather goes through a one-layer
LSTM once per case; soil and terrain through a linear layer; the last valid states feed an MLP head.
Both predict the standardised yield of one pixel.
"""
import math

import numpy as np
import torch
from torch import nn

from .data import DATE_PAD, NB, SCL_PAD


# ---------------------------------------------------------------------------------------------------------- Transformer
def date_encoding(embed_dim, pos, omega=None):
    pos = torch.where(pos == DATE_PAD, torch.zeros_like(pos), pos)
    if omega is None:
        omega = 1.0 / (10000 ** (torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float32) / (embed_dim / 2.0)))
    out = torch.einsum('bt,d->btd', pos, omega.to(device=pos.device))
    return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_size=128, dropout=0.0):
        super().__init__()
        self.norm_in = nn.LayerNorm(hidden_size, eps=1e-12)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm_out = nn.LayerNorm(hidden_size, eps=1e-12)
        self.ffn = nn.Sequential(nn.Linear(hidden_size, ff_size), nn.GELU(), nn.Linear(ff_size, hidden_size))

    def forward(self, x, pad_mask, return_att=False):
        x_norm = self.norm_in(x)
        attn_out, attn = self.attn(x_norm, x_norm, x_norm, key_padding_mask=pad_mask, need_weights=return_att,
                                   average_attn_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm_out(x))
        return x, attn if return_att else None


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, hidden_size, num_heads, num_layers):
        super().__init__()
        self.reg_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.embedding = nn.Linear(input_dim, hidden_size)
        omega = torch.arange(hidden_size // 2, dtype=torch.float32) / (hidden_size / 2.0)
        self.register_buffer('date_omega', 1.0 / (10000 ** omega), persistent=False)
        self.layers = nn.ModuleList([TransformerLayer(hidden_size, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, x, dates, return_att=False):
        hidden = self.embedding(x) + date_encoding(self.embedding.out_features, dates, self.date_omega)
        hidden = torch.cat([self.reg_token.expand(hidden.shape[0], -1, -1), hidden], dim=1)
        pad_mask = dates == DATE_PAD
        pad_mask = torch.cat([torch.zeros((pad_mask.shape[0], 1), dtype=torch.bool, device=pad_mask.device), pad_mask], dim=1)
        attns = []
        for layer in self.layers:
            hidden, attn = layer(hidden, pad_mask, return_att=return_att)
            if attn is not None:
                attns.append(attn)
        hidden = self.norm(hidden)
        return hidden[:, 0, :], tuple(attns)


class MultimodalTransformer(nn.Module):
    def __init__(self, s2_dim=NB + SCL_PAD + 1, weather_dim=4, soil_dim=24, dem_dim=5, hidden_size=32, num_heads=1, num_layers=4):
        super().__init__()
        self.s2_encoder = TemporalTransformer(s2_dim, hidden_size, num_heads, num_layers)
        self.weather_encoder = TemporalTransformer(weather_dim, hidden_size, num_heads, num_layers)
        self.dem_encoder = nn.Sequential(nn.Linear(dem_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, hidden_size))
        self.soil_encoder = nn.Sequential(nn.Linear(soil_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, hidden_size))
        self.regression = nn.Linear(4 * hidden_size, 1)

    def forward(self, batch, return_att=False):
        out_s2, att_s2 = self.s2_encoder(batch['x_s2'], batch['x_s2_dates'], return_att=return_att)
        out_w, att_w = self.weather_encoder(batch['x_weather'], batch['x_weather_dates'], return_att=return_att)
        z = torch.cat([out_s2, out_w, self.dem_encoder(batch['x_dem']), self.soil_encoder(batch['x_soil'])], dim=-1)
        pred = self.regression(z).squeeze(-1)
        return (pred, {'s2': att_s2, 'weather': att_w}) if return_att else pred


# ---------------------------------------------------------------------------------------------------------- 3D-LSTM
OFFSETS = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if (dr, dc) != (0, 0)]
CENTRE = 12
LSTM_FP32 = True


def neighbour_table(row, col, pix0):
    """[pixels, 24] index of each 5 x 5 neighbour inside the same field (-1 none) from the raster row / column of every pixel.

    pix0: field offsets into the pixel arrays (fields are consecutive), with the total as the last entry.
    """
    row, col, pix0 = (np.asarray(v, np.int64) for v in (row, col, pix0))
    fidx = np.repeat(np.arange(len(pix0) - 1), np.diff(pix0))
    span = int(max(row.max(), col.max())) + 8
    key = (fidx * span + (row + 4)) * span + (col + 4)
    order = np.argsort(key, kind='stable')
    ks = key[order]
    nb = np.full((len(row), len(OFFSETS)), -1, np.int64)
    for j, (dr, dc) in enumerate(OFFSETS):
        q = (fidx * span + (row + dr + 4)) * span + (col + dc + 4)
        pos = np.minimum(np.searchsorted(ks, q), len(ks) - 1)
        hit = ks[pos] == q
        nb[hit, j] = order[pos[hit]]
    return nb


def neighbour_samples(p, nb_static):
    """[samples, 24] neighbour sample index in p (-1 none); needs every pixel of a case as consecutive samples."""
    nbs = torch.as_tensor(nb_static, device=p.s_pix.device)[p.s_pix]
    i = torch.arange(p.n, device=p.s_pix.device)[:, None]
    cand = torch.where(nbs >= 0, i + (nbs - p.s_pix[:, None]), torch.full_like(nbs, -1))
    ok = (cand >= 0) & (cand < p.n)
    c = cand.clamp(0, p.n - 1)
    ok &= (p.s_case[c] == p.s_case[:, None]) & (p.s_pix[c] == nbs)
    assert int(((nbs >= 0) & ~ok).sum()) == 0, 'neighbour lookups outside their case'
    return torch.where(ok, cand, torch.full_like(cand, -1))


def _date_feats(d, ok):
    """[.., 3]: sine and cosine of the day of year and days since the first valid day / 365; zeros where not ok."""
    first = torch.where(ok, d, torch.full_like(d, float('inf'))).min(-1, keepdim=True).values
    first = torch.where(torch.isfinite(first), first, torch.zeros_like(first))
    a = 2 * math.pi * d / 365.25
    f = torch.stack([torch.sin(a), torch.cos(a), (d - first) / 365.0], -1)
    return torch.where(ok[..., None], f, torch.zeros_like(f))


def _compact(ok):
    n = ok.sum(1)
    Lc = max(int(n.max().item()) if len(n) else 1, 1)
    return torch.argsort((~ok).to(torch.int8), dim=1, stable=True)[:, :Lc], n, Lc


def batch3d(p, idx, st):
    """3D-LSTM batch: centre features at the valid steps, 5 x 5 neighbours, weather once per case, static, target.

    Needs p.nb_sample = neighbour_samples(p, ...).
    """
    dev = p.vals.device
    L = max(int(p.s_len[idx].max().item()) if len(idx) else 1, 1)
    ar = torch.arange(L, device=dev)
    inside = ar[None, :] < p.s_len[idx][:, None]
    g = (p.s_start[idx][:, None] + ar[None, :]).clamp(max=len(p.ok) - 1)
    ok = inside & p.ok[g]
    order, n, Lc = _compact(ok)
    keep = torch.arange(Lc, device=dev)[None, :] < n[:, None]
    gc = torch.gather(g, 1, order)
    x = (p.vals[gc] - st['s2'][0]) / st['s2'][1]
    x = torch.where(keep[..., None], x, torch.full_like(x, -1.0))
    scl = torch.where(keep, p.scl[gc].long(), torch.full_like(gc, SCL_PAD))
    oh = torch.nn.functional.one_hot(scl.clamp(0, SCL_PAD), SCL_PAD + 1).float()
    centre = torch.cat([x, oh, _date_feats(p.day[gc].float(), keep)], -1)
    nbs = p.nb_sample[idx]
    has = nbs >= 0
    step = gc - p.s_start[idx][:, None]
    gn = (p.s_start[nbs.clamp(min=0)][:, :, None] + step[:, None, :]).clamp(max=len(p.ok) - 1)
    okn = has[..., None] & keep[:, None, :] & p.ok[gn]
    xn = (p.vals[gn] - st['s2'][0]) / st['s2'][1]
    xn = torch.where(okn[..., None], xn, torch.full_like(xn, -1.0)).permute(0, 2, 1, 3)
    uc, inv = torch.unique(p.s_case[idx], return_inverse=True)
    wd = p.wd[uc]
    wok = (wd != DATE_PAD) & torch.isfinite(p.wv[uc]).all(-1)
    worder, wn, Lw = _compact(wok)
    wkeep = torch.arange(Lw, device=dev)[None, :] < wn[:, None]
    wv = (torch.gather(p.wv[uc], 1, worder[..., None].expand(-1, -1, 4)) - st['wx'][0]) / st['wx'][1]
    wv = torch.where(wkeep[..., None], torch.nan_to_num(wv, nan=-1.0), torch.full_like(wv, -1.0))
    weather = torch.cat([wv, _date_feats(torch.gather(wd, 1, worder), wkeep)], -1)
    soil = torch.nan_to_num((p.soil[p.s_pix[idx]] - st['soil'][0]) / st['soil'][1], nan=0.0)
    dem = torch.nan_to_num((p.dem[p.s_pix[idx]] - st['dem'][0]) / st['dem'][1], nan=0.0)
    y = (p.target[p.s_pix[idx]] - st['y'][0]) / st['y'][1]
    return {'centre': centre, 'x_nb': xn, 'n_valid': n, 'weather': weather, 'w_valid': wn, 'w_inv': inv,
            'static': torch.cat([soil, dem], -1), 'target': y}


def _last(out, n):
    return out[torch.arange(len(out), device=out.device), (n - 1).clamp(min=0)]


class Base3DLSTM(nn.Module):
    def __init__(self, bands=NB, scl_dim=SCL_PAD + 1, weather_dim=4, soil_dim=24, dem_dim=5, conv=64, hidden=64, w_hidden=32,
                 s_hidden=32, fc=64, layers=2, dropout=0.2):
        super().__init__()
        self.bands = bands
        self.spatial = nn.Sequential(nn.Linear(25 * bands, conv), nn.LeakyReLU())
        self.lstm = nn.LSTM(conv + scl_dim + 3, hidden, num_layers=layers, batch_first=True)
        self.wlstm = nn.LSTM(weather_dim + 3, w_hidden, num_layers=1, batch_first=True)
        self.static = nn.Sequential(nn.Linear(soil_dim + dem_dim, s_hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden + w_hidden + s_hidden, fc), nn.BatchNorm1d(fc), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(fc, 1))

    def patch(self, centre_bands, nb):
        return torch.cat([nb[:, :, :CENTRE], centre_bands[:, :, None], nb[:, :, CENTRE:]], 2).flatten(2)

    def forward(self, b):
        c = b['centre']
        z = self.spatial(self.patch(c[..., :self.bands], b['x_nb']))
        seq = torch.cat([z, c[..., self.bands:].to(z.dtype)], -1)
        with torch.autocast(device_type=seq.device.type, enabled=not LSTM_FP32):
            out, _ = self.lstm(seq.float() if LSTM_FP32 else seq)
            wout, _ = self.wlstm(b['weather'].float() if LSTM_FP32 else b['weather'].to(seq.dtype))
        h = _last(out, b['n_valid'])
        hw = _last(wout, b['w_valid'])[b['w_inv']].to(h.dtype)
        return self.head(torch.cat([h, hw, self.static(b['static']).to(h.dtype)], 1)).squeeze(1)
