"""Figure of the README from the outputs of the tutorial (public inputs and synthetic yields only): the new season's
predictions on a Sentinel-2 background (inference.png).

usage: python docs/tutorial_figures.py --run run --infer new --season <new season folder> --stage tau2 --out docs/images
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

YIELD = LinearSegmentedColormap.from_list('yield', ['#f3f0b5', '#b9d66b', '#5ea83e', '#1f6b2a', '#0b3d1c'])
DIV = 'RdBu_r'
DAY = 86_400_000
plt.rcParams.update({'font.size': 7, 'font.family': 'DejaVu Sans'})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--infer', type=Path, required=True)
    ap.add_argument('--season', type=Path, required=True)
    ap.add_argument('--stage', default='tau2')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    beta = json.loads((a.run / 'model/config.json').read_text())['strengths']['beta']
    P = pd.read_parquet(a.infer / 'pixels.parquet')
    X = pd.read_parquet(a.infer / 'cases.parquet')
    truth = pd.read_parquet(a.run / 'pixels.parquet')                    # synthetic yield, for evaluation only
    truth = truth[truth.case_id.isin(X.case_id)].sort_values(['case_id', 'pixel'])
    truth['k'] = truth.groupby('case_id').cumcount()
    P['k'] = P.groupby('case_id').cumcount()
    P = P.merge(truth[['case_id', 'k', 'truth']], on=['case_id', 'k'], how='left')
    st = X[X.stage == a.stage]
    pk = P[P.case_id.isin(st.case_id)]
    with np.load(a.season / 'collect/base_inputs/corn_grid.npz') as z:
        rs, cs = z['row'], z['col']
    raw = a.season / 'collect/raw/base_inputs/s2'
    meta = pd.DataFrame(json.loads((raw / 'IMAGES.json').read_text()))
    meta['day'] = meta.time_ms // DAY
    mid = int(st.decision_ms.median() // DAY)
    best, img, when = -1.0, None, None
    for name, day in meta.iloc[np.argsort(np.abs(meta.day.to_numpy() - mid))[:25]][['image_id', 'day']].itertuples(index=False):
        zz = np.load(raw / f'{name}.npz')                                 # the clearest of the 25 images nearest the stage
        clear = float(np.isin(zz['SCL'], (4, 5)).mean())
        if clear > best:
            best, img, when = clear, zz, day
    bg = np.clip(np.stack([img['B4'], img['B3'], img['B2']], -1).astype(float) / 2500.0, 0, 1)
    pix = pk.pixel.to_numpy()
    r0, r1 = max(rs[pix].min() - 12, 0), rs[pix].max() + 12
    c0, c1 = max(cs[pix].min() - 12, 0), cs[pix].max() + 12
    ymin, ymax = np.nanpercentile(np.r_[pk.base, pk.truth], [2, 98])
    corr = beta * pk.s15.to_numpy()
    smax = max(np.nanpercentile(np.abs(corr), 98), 1e-6)
    aspect = (r1 - r0) / (c1 - c0)
    fig, axes = plt.subplots(1, 4, figsize=(7.5, 1.85 * aspect + 1.1), dpi=300)
    for ax, (v, title, cm, kw) in zip(axes, ((pk.base, 'Base', YIELD, dict(vmin=ymin, vmax=ymax)),
                                             (corr, f'Within-field correction β·S15 (β = {beta:g})', DIV,
                                              dict(norm=TwoSlopeNorm(0, -smax, smax))),
                                             (pk.fieldwise, 'FieldWise', YIELD, dict(vmin=ymin, vmax=ymax)),
                                             (pk.truth, 'Synthetic yield (evaluation only)', YIELD, dict(vmin=ymin, vmax=ymax)))):
        ax.imshow(bg[r0:r1, c0:c1], interpolation='nearest')
        A = np.full(bg.shape[:2], np.nan)
        A[rs[pix], cs[pix]] = np.asarray(v, float)
        im = ax.imshow(A[r0:r1, c0:c1], cmap=cm, interpolation='nearest', **kw)
        ax.set_title(title, fontsize=6.8)
        ax.set_axis_off()
        cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.03)
        cb.ax.tick_params(labelsize=5.5)
        cb.set_label('t/ha', fontsize=6)
    h, w = A[r0:r1, c0:c1].shape
    axes[0].plot([w - 32, w - 12], [h - 6, h - 6], color='white', lw=1.5)
    axes[0].text(w - 22, h - 8, '200 m', color='white', fontsize=5.5, ha='center', va='bottom')
    axes[0].set_xlim(-0.5, w - 0.5)
    axes[0].set_ylim(h - 0.5, -0.5)
    y = int(X.year.iloc[0]) - 1
    fig.suptitle(f'New season {y}/{(y + 1) % 100:02d}, {a.stage.replace("tau", "τ")}: {len(st)} image-screened areas in Tainan, '
                 f'predicted without labels. Background: Sentinel-2, {pd.to_datetime(when, unit="D").date()}', fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(a.out / 'inference.png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
