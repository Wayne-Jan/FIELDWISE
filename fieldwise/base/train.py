"""Training and prediction of the Base models (manuscript Section 4.1, Fig. 3).

Supports both the whole-season strategy (trained and selected on full seasons) and the truncated-season strategy (per-epoch
random cutoff truncation via truncation.CaseClock and checkpoint selection across five observation windows via dev_pack).
Recipe: AdamW (learning rate 1e-3, weight decay 0.02), batch 2,048, gradient clipping at 1, at most 55 epochs with 10%
linear warm-up and cosine decay, four validation checks per epoch plus one at the epoch end, patience 12 checks. Training
uses BF16 autocast on GPU, with the Transformer date encoding and the 3D-LSTM recurrent layers in FP32; validation and
prediction run in FP32.
"""
import json
import math

import numpy as np
import torch

from . import data as D
from .models import Base3DLSTM, MultimodalTransformer, batch3d

RECIPE = dict(epochs=55, checks_per_epoch=4, patience=12, batch=2048, lr=1e-3, weight_decay=0.02, warmup=0.1, clip=1.0)
SEEDS = (42, 43, 44, 45, 46)


def build(arch, soil_dim, dem_dim):
    """(model, batch function) of an architecture: 'transformer' or '3dlstm'."""
    if arch == 'transformer':
        return MultimodalTransformer(soil_dim=soil_dim, dem_dim=dem_dim), D.batch
    if arch == '3dlstm':
        return Base3DLSTM(soil_dim=soil_dim, dem_dim=dem_dim), batch3d
    raise ValueError(arch)


def dev_split(groups, fraction=0.15, seed=42):
    """Development groups: a random fraction of the field groups (whole fields, so that the same land never splits)."""
    groups = sorted(set(groups))
    perm = np.random.default_rng(seed).permutation(len(groups))
    return {groups[k] for k in perm[:max(1, int(round(fraction * len(groups))))]}


@torch.no_grad()
def predict(model, p, idx, st, batch_fn, batch=2 * RECIPE['batch']):
    """FP32 predictions (t/ha) of samples idx, shortest sequences first."""
    model.eval()
    out = torch.empty(len(idx), device=p.vals.device)
    order = torch.argsort(p.s_len[idx])
    for s in range(0, len(idx), batch):
        k = order[s:s + batch]
        out[k] = model(batch_fn(p, idx[k], st)).float()
    return out * st['y'][1] + st['y'][0]


def mse_z(pred, p, idx, st):
    y, m = p.target[p.s_pix[idx]], p.gt_valid[p.s_pix[idx]]
    return float((((pred - y) / st['y'][1])[m] ** 2).mean()) if bool(m.any()) else float('nan')


def train_model(arch, p, fit_idx, dev_idx, seed, recipe=RECIPE, trunc=None, dev_pack=None):
    """Train one model; returns (model at its best check, standardisation statistics, history, best epoch).

    When trunc (truncation.CaseClock) is given, each epoch truncates fit samples at per-field-season random cutoffs.
    When dev_pack is given, validation runs on dev_pack[dev_idx] (e.g. all five observation windows of the dev split).
    """
    dev = p.vals.device
    p_dev = p if dev_pack is None else dev_pack
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sel = torch.zeros(p.n, dtype=torch.bool, device=dev)
    sel[fit_idx] = True
    st = D.fit_stats(p, sel)
    model, batch_fn = build(arch, p.soil.shape[1], p.dem.shape[1])
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=recipe['lr'], weight_decay=recipe['weight_decay'])
    per_epoch = math.ceil(len(fit_idx) / recipe['batch'])
    total = recipe['epochs'] * per_epoch
    warm = max(1, int(total * recipe['warmup']))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    lengths = p.s_len[fit_idx].cpu().numpy()
    every = max(1, per_epoch // recipe['checks_per_epoch'])
    hist = []
    S = dict(best=float('inf'), pbest=float('inf'), bad=0, best_step=0, state=None, last_check=-1)
    step, stop = 0, False

    def check(kind):
        v = mse_z(predict(model, p_dev, dev_idx, st, batch_fn), p_dev, dev_idx, st)
        model.train()
        rec = dict(kind=kind, step=step, epoch=round(step / per_epoch, 4), dev_mse_z=v)
        if v < S['best'] - 1e-7:
            S.update(best=v, best_step=step, state={k: x.detach().clone() for k, x in model.state_dict().items()})
        if kind == 'periodic':
            if v < S['pbest'] - 1e-7:
                S.update(pbest=v, bad=0)
            else:
                S['bad'] += 1
            S['last_check'] = step
        hist.append(rec)

    for ep in range(1, recipe['epochs'] + 1):
        model.train()
        pv, lengths_ep = p, lengths
        if trunc is not None:
            pv, _ = trunc.view(seed, ep)
            lengths_ep = pv.s_len[fit_idx].cpu().numpy()
        for pos, b_local in enumerate(D.bucket_batches(lengths_ep, recipe['batch'], rng), 1):
            idx = fit_idx[torch.from_numpy(b_local).to(dev)]
            b = batch_fn(pv, idx, st)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
                pred = model(b)
            loss = torch.nn.functional.mse_loss(pred.float(), b['target'])
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe['clip'])
            opt.step()
            sched.step()
            step += 1
            if pos % every == 0 and pos // every <= recipe['checks_per_epoch']:
                check('periodic')
                if S['bad'] >= recipe['patience']:
                    stop = True
        if S['last_check'] != step:
            check('epoch_end')
        if stop:
            break
    if S['state'] is not None:
        model.load_state_dict(S['state'])
    return model, st, hist, S['best_step'] / per_epoch


def save(model, st, folder):
    """model.pt (state dict) and STATS.json (standardisation statistics)."""
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder / 'model.pt')
    (folder / 'STATS.json').write_text(json.dumps({k: [x.cpu().tolist() for x in v] for k, v in st.items()}) + '\n')


def load(arch, folder, soil_dim, dem_dim, device='cpu'):
    """(model, batch function, statistics) of a saved model."""
    model, batch_fn = build(arch, soil_dim, dem_dim)
    model.load_state_dict(torch.load(folder / 'model.pt', map_location=device, weights_only=True))
    st = {k: tuple(torch.tensor(x, device=device) for x in v) for k, v in json.loads((folder / 'STATS.json').read_text()).items()}
    return model.to(device).eval(), batch_fn, st
