"""Tutorial run: synthetic yields -> rolling Base -> RPT database -> FieldWise for every year -> the target season.

Every fit serving year T uses only history legal at T's cutoff (selection.py): the Base of year T is trained on the
harvest-time cases of earlier seasons whose labels are available by the cutoff, and its predictions of year T are the
out-of-sample predictions that enter the residual bank. The target season's labels are held out of every fit and read only
for the evaluation at the end.

usage: python -m fieldwise.tutorial.run --seasons <season folders, oldest first> --target <harvest year> --out <folder>
       [--seeds 42] [--epochs 55] [--device cuda]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import bank as fb, calibration, decision, environment, readout, retrieval, rpt, selection, strength
from ..base import data as BD, train as BT
from ..readout import b7_features
from . import model as model_io
from .study import NAMED, Study
from .synth import generate

STAGES = ('tau1', 'tau2', 'tau3', 'tau4')
QGROUPS = {'s2': ('q_s2_fixedgdd4__', 'q_s2_asof__'), 'weather': ('q_weather_fixedgdd4__',), 'soil': ('q_soil__',),
           'topo': ('q_topo__',)}


def rolling_base(S, target_fit, gt_fit, cut, seeds, recipe, device, log, target, save_dir):
    """{case_id: Base prediction of every service pixel (mean over seeds)} for every year after the first; the models of
    the target year are saved to save_dir/seed_<n>."""
    C, H = S.cases, S.cases[S.cases.stage == 'harvest']
    pred = {}
    for T in sorted(C.year.unique())[1:]:
        tr = H[(H.year < T) & (H.label_ms <= cut[T])].reset_index(drop=True)
        if tr.empty:
            continue
        t0 = time.time()
        p_tr = BD.PixelData(S.case_inputs(tr), S.soil, S.dem, target_fit, gt_fit, samples='gt', device=device)
        key = (tr.year.astype(str) + ':' + tr.fid.astype(str)).to_numpy()[p_tr.s_case.cpu().numpy()]
        q = C[C.year == T].reset_index(drop=True)
        p_q = BD.PixelData(S.case_inputs(q), S.soil, S.dem, target_fit, gt_fit, samples='all', device=device)
        out, runs = 0., []
        for seed in seeds:
            dev = BT.dev_split(key, 0.15, seed)
            is_dev = torch.from_numpy(np.isin(key, list(dev))).to(device)
            idx = torch.arange(p_tr.n, device=device)
            model, st, hist, best = BT.train_model('transformer', p_tr, idx[~is_dev], idx[is_dev], seed, recipe)
            out = out + BT.predict(model, p_q, torch.arange(p_q.n, device=device), st, BD.batch)
            if T == target:
                BT.save(model, st, save_dir / f'seed_{seed}')
            runs.append(dict(seed=seed, best_epoch=round(best, 2), checks=len(hist), dev_mse_z=min(h['dev_mse_z'] for h in hist)))
        yhat = (out / len(seeds)).cpu().numpy()
        s_case = p_q.s_case.cpu().numpy()
        for i, cid in enumerate(q.case_id):
            pred[cid] = yhat[s_case == i]
        log[int(T)] = dict(train_field_seasons=len(tr), train_pixels=int(p_tr.n), query_cases=len(q), runs=runs,
                           seconds=round(time.time() - t0, 1))
    return pred


def save_database(folder, bank, B, S, Q, target):
    """RPT database of the labelled history (cases before the target year), self-contained for tutorial.infer: residual
    bank, pixel embeddings and their index, retrieval descriptors, and pixel coordinates."""
    hist = (B.year < target).to_numpy()
    keep = hist[bank.row]
    remap = np.cumsum(hist) - 1
    H = fb.Bank(bank.cases[hist].reset_index(drop=True), remap[bank.row[keep]], bank.pixel[keep], bank.pred[keep],
                bank.target[keep], bank.gt_valid[keep])
    H.save(folder, 'bank')
    ix = S.emb_index.set_index('case_id').loc[H.cases.case_id]
    rows = np.concatenate([np.arange(s, s + n) for s, n in zip(ix.row_start, ix.n_pixels)])
    np.save(folder / 'embeddings.npy', S.emb[rows])
    pd.DataFrame(dict(case_id=H.cases.case_id, row_start=np.r_[0, np.cumsum(ix.n_pixels.to_numpy())[:-1]],
                      n_pixels=ix.n_pixels.to_numpy())).to_parquet(folder / 'embedding_index.parquet', index=False)
    Q[hist].reset_index().to_parquet(folder / 'descriptors.parquet', index=False)
    np.savez(folder / 'pixels_lonlat.npz', pixel=H.pixel, lon=S.lon[H.pixel], lat=S.lat[H.pixel])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seasons', nargs='+', required=True)
    ap.add_argument('--target', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--epochs', type=int, default=BT.RECIPE['epochs'])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--labels', type=Path, help='your yield maps instead of synthetic yields: npz with target [pixels] (t/ha) '
                                               'and gt_valid [pixels], in the pixel order of the seasons given')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    rec = dict(seasons=a.seasons, target=a.target, seeds=a.seeds, epochs=a.epochs, device=a.device, timing={},
               labels=str(a.labels) if a.labels else 'synthetic')
    S = Study(a.seasons)
    C = S.cases

    # yields (synthetic unless --labels); the target season's labels are held out of every fit
    if a.labels:
        z = np.load(a.labels)
        target, gt = z['target'].astype(np.float32), z['gt_valid'].astype(bool) & np.isfinite(z['target'])
        assert len(target) == S.n_pix, f'labels cover {len(target)} pixels, the seasons {S.n_pix}'
    else:
        target, gt, comp, FS = generate(S)
        comp.to_parquet(a.out / 'synthetic_components.parquet', index=False)
        FS.to_parquet(a.out / 'synthetic_field_seasons.parquet', index=False)
    hold = np.zeros(S.n_pix, bool)
    _, tp = S.pixel_index(C[(C.year == a.target) & (C.stage == 'harvest')])
    hold[tp] = True
    target_fit, gt_fit = np.where(hold, np.nan, target).astype(np.float32), gt & ~hold
    np.savez_compressed(a.out / 'yield_history.npz', target=target_fit, gt_valid=gt_fit)
    np.savez_compressed(a.out / 'yield_target_heldout.npz', pixels=tp, target=target[tp], gt_valid=gt[tp])
    rec['timing']['synthetic'] = round(time.time() - t_all, 1)

    # cutoffs from all queries, labelled or not
    table = selection.cutoff_table(C[['year', 'stage', 'decision_ms']])
    cut = selection.phase_cutoffs(table, 'in_season')

    # 1. rolling Base: out-of-sample predictions of every year after the first
    t0 = time.time()
    rec['base'] = {}
    recipe = dict(BT.RECIPE, epochs=a.epochs)
    pred = rolling_base(S, target_fit, gt_fit, cut, a.seeds, recipe, a.device, rec['base'], a.target, a.out / 'model/base')
    rec['timing']['base'] = round(time.time() - t0, 1)

    # 2. RPT database: residual bank of the in-season cases with out-of-sample predictions, references, embeddings, geometry
    t0 = time.time()
    B = C[C.stage.isin(STAGES) & C.case_id.isin(pred)].reset_index(drop=True)
    row, pix = S.pixel_index(B)
    bank = fb.Bank(B[['case_id', 'year', 'field', 'stage', 'decision_ms', 'label_ms']].assign(
                       pix0=B.gpix0.to_numpy(), n_service=B.n_pix.to_numpy()),
                   row, pix, np.concatenate([pred[c] for c in B.case_id]).astype(np.float32), target_fit[pix], gt_fit[pix])
    M = bank.moments()
    bank.cases['rbar'] = M.rbar.where(M.n > 0).to_numpy()                       # field-mean residual read by RPT
    Q = S.q.reindex(B.case_id)
    num = [c for c in Q.columns if pd.api.types.is_numeric_dtype(Q[c])]
    refs = retrieval.select_references(bank.cases, Q[num].reset_index(drop=True), retrieval.prefix_groups(num, QGROUPS), k=50)
    geometry = {}
    for (y, f), g in B.groupby(['year', 'fid']):                                 # one neighbour list per field-season
        p0, n = int(g.gpix0.iloc[0]), int(g.n_pix.iloc[0])
        nb = rpt.neighbours(S.lon[p0:p0 + n], S.lat[p0:p0 + n])
        geometry.update({c: nb for c in g.case_id})
    save_database(a.out / 'rpt_database', bank, B, S, Q[num], a.target)
    rec['rpt_database'] = dict(cases=len(B), pixels=int(len(pix)), labelled_pixels=int(gt_fit[pix].sum()),
                               reference_pairs=len(refs), queries_with_references=int(refs.query_row.nunique()))
    R = rpt.RPT(bank, S.emb, S.emb_index, refs, geometry, device=a.device)
    years = sorted(B.year.unique())
    sums, s15, rec['rpt'], rpt_models = [pd.DataFrame(columns=['row', 'n', 'sr', 'srr', 'sS', 'srS', 'sSS'], dtype=float)], {}, {}, {}
    for T in sorted(set(B.year.to_numpy()[R.queries])):                         # years whose cases have references
        model, log, sm, px = rpt.run_year(R, T, cutoff=cut)
        rpt_models[T] = model
        sums.append(sm)
        s15.update(px)
        rec['rpt'][int(T)] = {k: v for k, v in log.items() if k in ('dev_years', 'dev_queries', 'epochs', 'tau')}
    SM = pd.concat(sums, ignore_index=True).astype(float).set_index('row').reindex(range(len(B)))
    rec['timing']['rpt'] = round(time.time() - t0, 1)

    # 3. G, K, readouts, strengths and alpha (in-season phase cutoffs)
    t0 = time.time()
    m6 = np.column_stack([M.n, M.sr, M.srr] + [SM[k].fillna(0).to_numpy() for k in ('sS', 'srS', 'sSS')]).astype(float)
    X = B.join(M[['rbar', 'mse', 'n', 'sy', 'sp', 'spp', 'syp', 'p_mean', 'p_sd', 'log_n']])
    X.loc[X.n == 0, ['rbar', 'mse']] = np.nan
    X['w'] = selection.field_season_weights(X)
    named = np.column_stack([np.column_stack([(X.stage == t).to_numpy(float) for t in STAGES]),
                             X[['p_mean', 'p_sd', 'log_n']].to_numpy(float), S.named.reindex(X.case_id)[list(NAMED)].to_numpy(float)])
    lag = environment.lagged(S.daily, X.fs_row.to_numpy(), X.n_days.to_numpy())
    G, env_models, grec = environment.fit_environment(X, named, lag, S.daily, S.fs.year.to_numpy(), S.fs.start_day.to_numpy(), cutoff=cut)
    lab = X[X.n > 0]
    coef = calibration.fit_affine(lab, cutoff=cut, targets=years)
    Cal = calibration.calibration_offset(coef, X.year, X.stage, X.p_mean)
    b7 = b7_features(X.p_mean.to_numpy(), X.p_sd.to_numpy(), X.log_n.to_numpy(), G, Cal)
    rh, ro_models = [], {}
    for j, fm in enumerate(S.fm_models):
        means = np.stack([S.fm_means[c][:, j] for c in X.case_id]).astype(np.float64)
        r, models = readout.readouts(X, b7, means, np.full(len(X), 2), X.rbar.to_numpy(), cutoff=cut)
        rh.append(r)
        ro_models[fm] = {s: m for (s, T), m in models.items() if T == a.target}
    with np.errstate(all='ignore'):
        rhat = np.nanmean(np.column_stack(rh), 1)
    ok = (X.n > 0).to_numpy()
    choice, _ = strength.select_strengths(lab, m6[ok], G[ok], Cal[ok], cutoff=cut, targets=years)
    gamma, beta, kappa = (X.year.map({T: v[i] for T, v in choice.items()}).to_numpy(float) for i in range(3))
    U, c = decision.offsets(rhat, G, Cal, gamma, kappa)
    with np.errstate(all='ignore'):
        L = decision.alpha_losses(m6, U, c, beta)
    alpha = decision.choose_alpha(X, L, ok, cutoff=cut, targets=years)
    u = U[np.arange(len(X)), X.year.map(alpha).to_numpy()]
    rec['timing']['correction'] = round(time.time() - t0, 1)
    tm = model_io.save(a.out / 'model', a.target, STAGES, S.fm_models, coef[coef.year == a.target], choice[a.target],
                       decision.ALPHAS[alpha[a.target]], env_models.get(a.target), ro_models, rpt_models.get(a.target))
    rec['choices'] = {int(T): dict(gamma=v[0], beta=v[1], kappa=v[2], alpha=float(decision.ALPHAS[alpha[T]]),
                                   G_fitted=bool(np.isfinite(G[X.year == T]).any()),
                                   Cal_fitted=bool(np.isfinite(Cal[X.year == T]).any()),
                                   readout_cases=int(np.isfinite(rhat[X.year == T]).sum()), cases=int((X.year == T).sum()))
                      for T, v in choice.items()}

    # 4. predictions and evaluation (the target season's synthetic labels are read only here)
    truth = target.copy()
    pix_rows = []
    for i, r in enumerate(X.itertuples()):
        a0, b0 = bank.bounds[i], bank.bounds[i + 1]
        base = bank.pred[a0:b0].astype(float)
        s = s15.get(i)
        s = np.zeros(len(base)) if s is None else np.asarray(s, float)
        pix_rows.append(pd.DataFrame(dict(case_id=r.case_id, year=r.year, stage=r.stage, pixel=pix[a0:b0], base=base,
                                          s15=s, correction_only=base + beta[i] * s + c[i], fieldwise=base + beta[i] * s + u[i],
                                          truth=truth[pix[a0:b0]], labelled=gt[pix[a0:b0]])))
    P = pd.concat(pix_rows, ignore_index=True)
    P['lon'], P['lat'] = S.lon[P.pixel], S.lat[P.pixel]
    P.to_parquet(a.out / 'pixels.parquet', index=False)
    lp = P[P.labelled]
    err = lp.assign(**{k: (lp[k] - lp.truth) ** 2 for k in ('base', 'correction_only', 'fieldwise')})
    ev = err.groupby(['year', 'case_id']).agg(base=('base', 'mean'), correction_only=('correction_only', 'mean'),
                                              fieldwise=('fieldwise', 'mean')).reset_index()
    rec['evaluation_mse_by_year'] = ev.groupby('year')[['base', 'correction_only', 'fieldwise']].mean().round(4).to_dict('index')
    X.assign(G=G, Cal=Cal, rhat=rhat, c=c, u=u, beta=beta).to_parquet(a.out / 'cases.parquet')
    tq = X.index[X.year == a.target]
    refs[refs.query_row.isin(tq)].assign(query=lambda d: X.case_id.to_numpy()[d.query_row],
                                         reference=lambda d: X.case_id.to_numpy()[d.donor_row]).to_parquet(
        a.out / 'target_references.parquet', index=False)
    rec['timing']['total'] = round(time.time() - t_all, 1)
    (a.out / 'RUN.json').write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: rec[k] for k in ('timing', 'rpt_database', 'choices', 'evaluation_mse_by_year')}, indent=1, default=str))


if __name__ == '__main__':
    main()
