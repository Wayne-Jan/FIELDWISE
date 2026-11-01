"""FieldWise inference of a new season from a saved model and an RPT database, without labels and without refitting.

Inputs: the model folder and the RPT database written by tutorial.run, and the collected season (collect/, encode/). Steps:
Base (mean of the saved seeds) -> G, Cal, readouts -> decision offset u -> references from the database -> within-field
correction S15 -> Base, Correction only and FieldWise per pixel.

usage: python -m fieldwise.tutorial.infer --model run/model --database run/rpt_database --season <season folder> --out <folder>
       [--check run/pixels.parquet] [--device cuda]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import bank as fb, decision, environment, readout, retrieval, rpt
from ..base import data as BD, train as BT
from ..readout import b7_features
from . import model as model_io
from .run import QGROUPS, STAGES
from .study import NAMED, Study


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--database', type=Path, required=True)
    ap.add_argument('--season', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--check', type=Path)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    M = model_io.load(a.model)
    cfg = M['config']
    S = Study([a.season])
    X = S.cases[S.cases.stage.isin(STAGES)].reset_index(drop=True)
    n = len(X)

    # Base: mean of the saved seeds over every service pixel
    p = BD.PixelData(S.case_inputs(X), S.soil, S.dem, np.full(S.n_pix, np.nan, np.float32), np.zeros(S.n_pix, bool),
                     samples='all', device=a.device)
    idx = torch.arange(p.n, device=a.device)
    seeds = []
    for d in M['base_dirs']:
        model, batch_fn, st = BT.load('transformer', d, S.soil.shape[1], S.dem.shape[1], a.device)
        seeds.append(BT.predict(model, p, idx, st, batch_fn).cpu().numpy())
    pred = np.mean(seeds, 0).astype(np.float32)
    row, pix = S.pixel_index(X)
    p64 = pred.astype(np.float64)
    cnt = np.bincount(row, minlength=n)
    p_mean = np.bincount(row, weights=p64, minlength=n) / cnt
    p_sd = np.sqrt(np.maximum(np.bincount(row, weights=p64 * p64, minlength=n) / cnt - p_mean ** 2, 0))
    log_n = np.log(cnt)

    # field-mean components and the decision layer
    named = np.column_stack([np.column_stack([(X.stage == t).to_numpy(float) for t in STAGES]), p_mean, p_sd, log_n,
                             S.named.reindex(X.case_id)[list(NAMED)].to_numpy(float)])
    lag = environment.lagged(S.daily, X.fs_row.to_numpy(), X.n_days.to_numpy())
    G = M['environment'].predict(named, lag) if M['environment'] is not None else np.full(n, np.nan)
    ab = pd.DataFrame(cfg['calibration'], index=['a', 'b']).T.reindex(X.stage)
    Cal = (ab.a + (ab.b - 1) * p_mean).to_numpy()
    b7 = b7_features(p_mean, p_sd, log_n, G, Cal)
    rh = np.full((n, len(cfg['foundation_models'])), np.nan)
    for j, fm in enumerate(cfg['foundation_models']):
        means = np.stack([S.fm_means[c][:, j] for c in X.case_id]).astype(np.float64)
        full, first_only = readout.window_blocks(means, np.full(n, 2))
        for s, m in M['readouts'][fm].items():
            q = (X.stage == s).to_numpy()
            rh[q, j] = m.predict(b7[q], (first_only if s == STAGES[0] else full)[q])
    with np.errstate(all='ignore'):
        rhat = np.nanmean(rh, 1)
    k = cfg['strengths']
    U, c = decision.offsets(rhat, G, Cal, np.full(n, k['gamma']), np.full(n, k['kappa']))
    u = U[:, int(np.flatnonzero(np.isclose(decision.ALPHAS, cfg['alpha']))[0])]

    # RPT: the new season's cases join the database as queries without labels
    H = fb.Bank.load(a.database, 'bank')
    off = int(H.pixel.max()) + 1
    Xq = X[['case_id', 'year', 'field', 'stage', 'decision_ms', 'label_ms']].assign(
        pix0=off + X.gpix0.to_numpy(), n_service=X.n_pix.to_numpy(), rbar=np.nan)
    cases = pd.concat([H.cases, Xq], ignore_index=True)
    bank = fb.Bank(cases, np.r_[H.row, len(H.cases) + row], np.r_[H.pixel, off + pix], np.r_[H.pred, pred],
                   np.r_[H.target, np.full(len(pix), np.nan, np.float32)], np.r_[H.gt_valid, np.zeros(len(pix), bool)])
    ih = pd.read_parquet(a.database / 'embedding_index.parquet')
    it = S.emb_index.set_index('case_id').loc[X.case_id]
    rows = np.concatenate([np.arange(s, s + m) for s, m in zip(it.row_start, it.n_pixels)])
    eh = np.load(a.database / 'embeddings.npy')
    emb = np.concatenate([eh, S.emb[rows]])
    index = pd.concat([ih, pd.DataFrame(dict(case_id=X.case_id, n_pixels=it.n_pixels.to_numpy(),
                                             row_start=len(eh) + np.r_[0, np.cumsum(it.n_pixels.to_numpy())[:-1]]))],
                      ignore_index=True)
    Qh = pd.read_parquet(a.database / 'descriptors.parquet').set_index('case_id')
    Q = pd.concat([Qh, S.q.reindex(X.case_id)[Qh.columns]]).loc[cases.case_id].reset_index(drop=True)
    refs = retrieval.select_references(cases, Q, retrieval.prefix_groups(list(Q.columns), QGROUPS), k=50)
    geometry = {}
    for f, g in X.groupby('fid'):                                                # field-seasons of the new season
        p0, m = int(g.gpix0.iloc[0]), int(g.n_pix.iloc[0])
        nb = rpt.neighbours(S.lon[p0:p0 + m], S.lat[p0:p0 + m])
        geometry.update({cid: nb for cid in g.case_id})
    s15, s_raw = np.zeros(len(pix)), np.zeros(len(pix))
    if M['rpt'] is not None:
        R = rpt.RPT(bank, emb, index, refs, geometry, device=a.device)
        bounds = np.r_[0, np.cumsum(cnt)]
        for q in R.queries[R.queries >= len(H.cases)]:
            Sq, S15 = R.infer(int(q), M['rpt'])
            i = int(q) - len(H.cases)
            s_raw[bounds[i]:bounds[i + 1]] = Sq.cpu().numpy()
            s15[bounds[i]:bounds[i + 1]] = (S15 if S15 is not None else Sq).cpu().numpy()

    beta = k['beta']
    P = pd.DataFrame(dict(case_id=X.case_id.to_numpy()[row], stage=X.stage.to_numpy()[row], pixel=pix, lon=S.lon[pix],
                          lat=S.lat[pix], base=pred.astype(float), s_raw=s_raw, s15=s15,
                          correction_only=pred + beta * s15 + c[row], fieldwise=pred + beta * s15 + u[row]))
    P.to_parquet(a.out / 'pixels.parquet', index=False)
    X.assign(G=G, Cal=Cal, rhat=rhat, c=c, u=u, u_shrink=U[:, 0], u_reweighted=U[:, -1],
             **{f'rhat_{fm}': rh[:, j] for j, fm in enumerate(cfg['foundation_models'])}).to_parquet(a.out / 'cases.parquet')
    tq = refs[refs.query_row >= len(H.cases)]
    tq.assign(query=cases.case_id.to_numpy()[tq.query_row], reference=cases.case_id.to_numpy()[tq.donor_row]).to_parquet(
        a.out / 'references.parquet', index=False)
    rep = dict(cases=n, pixels=len(P), queries_with_references=int(tq.query_row.nunique()))
    if a.check:
        ref = pd.read_parquet(a.check)
        ref = ref[ref.case_id.isin(X.case_id)].sort_values(['case_id', 'pixel']).reset_index(drop=True)
        mine = P.assign(k=P.groupby('case_id').cumcount()).sort_values(['case_id', 'k']).reset_index(drop=True)
        rep['check_max_abs_diff'] = {c2: float(np.abs(mine[c2].to_numpy() - ref[c2].to_numpy()).max())
                                     for c2 in ('base', 's15', 'correction_only', 'fieldwise')}
    (a.out / 'INFER.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
