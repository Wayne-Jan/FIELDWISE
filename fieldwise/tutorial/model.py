"""Saved FieldWise model of one target year (written by tutorial.run, read by tutorial.infer).

<folder>/config.json      target year, stages, foundation models, calibration (a, b per stage), strengths, alpha
<folder>/environment.npz  G of the target year (environment.YearModel), when fitted
<folder>/readout_<fm>.npz readout of every stage that has one (readout.ReadoutModel)
<folder>/rpt.npz          RPT projection A, temperature tau and initial A0, when fitted
<folder>/base/seed_<n>/   Base model of every seed (model.pt, STATS.json)
"""
import json
from pathlib import Path

import numpy as np

from .. import environment, readout, rpt


def save(folder, target, stages, fm_models, coef, strengths, alpha, env_model, readouts, rpt_model):
    d = Path(folder)
    d.mkdir(parents=True, exist_ok=True)
    cfg = dict(target_year=int(target), stages=list(stages), foundation_models=list(fm_models),
               calibration={r.stage: [float(r.a), float(r.b)] for r in coef.itertuples()},
               strengths=dict(gamma=float(strengths[0]), beta=float(strengths[1]), kappa=float(strengths[2])), alpha=float(alpha),
               readout_stages={fm: sorted(m) for fm, m in readouts.items()})
    (d / 'config.json').write_text(json.dumps(cfg, indent=1))
    if env_model is not None:
        e = env_model
        np.savez(d / 'environment.npz', pair=np.asarray(e.pair, float), daily_mean=e.daily_scaler[0], daily_sd=e.daily_scaler[1],
                 col_mean=e.scaler[0], col_sd=e.scaler[1], col_has=e.scaler[2], x_mean=e.x_mean, y_mean=e.y_mean, beta=e.beta)
    for fm, models in readouts.items():
        arrays = {}
        for s, m in models.items():
            (bm, bs, bk), (fmn, fs, fk) = m.scalers
            arrays.update({f'{s}_b_mean': bm, f'{s}_b_sd': bs, f'{s}_b_keep': bk, f'{s}_f_mean': fmn, f'{s}_f_sd': fs,
                           f'{s}_f_keep': fk, f'{s}_y_mean': m.y_mean, f'{s}_beta': m.beta, f'{s}_lambdas': np.asarray(m.lambdas)})
        np.savez(d / f'readout_{fm}.npz', **arrays)
    if rpt_model is not None:
        np.savez(d / 'rpt.npz', A=rpt_model.A, tau=rpt_model.tau, A0=rpt_model.A0)


def load(folder):
    d = Path(folder)
    cfg = json.loads((d / 'config.json').read_text())
    env = None
    if (d / 'environment.npz').exists():
        z = np.load(d / 'environment.npz')
        env = environment.YearModel(cfg['target_year'], tuple(z['pair']), (z['daily_mean'], z['daily_sd']),
                                    (z['col_mean'], z['col_sd'], z['col_has']), z['x_mean'], float(z['y_mean']), z['beta'])
    readouts = {}
    for fm in cfg['foundation_models']:
        z = np.load(d / f'readout_{fm}.npz')
        readouts[fm] = {s: readout.ReadoutModel(((z[f'{s}_b_mean'], z[f'{s}_b_sd'], z[f'{s}_b_keep']),
                                                 (z[f'{s}_f_mean'], z[f'{s}_f_sd'], z[f'{s}_f_keep'])),
                                                float(z[f'{s}_y_mean']), z[f'{s}_beta'], tuple(z[f'{s}_lambdas']))
                        for s in cfg['readout_stages'][fm]}
    rm = None
    if (d / 'rpt.npz').exists():
        z = np.load(d / 'rpt.npz')
        rm = rpt.Model(z['A'], float(z['tau']), z['A0'])
    return dict(config=cfg, environment=env, readouts=readouts, rpt=rm, base_dirs=sorted((d / 'base').glob('seed_*')))
