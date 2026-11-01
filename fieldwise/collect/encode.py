"""Run the four official encoders on a collected season (reader -> views -> encoder).

Outputs:
  - TESSERA nano per-pixel representations of every case's own window (all stages), the RPT input: float16 [rows, 128] in
    case order and field pixel order, with an index (case_id, row_start, n_pixels);
  - field means of the four FMs (TESSERA, Galileo, OlmoEarth nano; Presto default) over three windows per case: the calendar
    year before sowing (year of the season start proxy), the first decision window (tau1) and the case's own window.
Each window is encoded once per model and reused by every case that needs it. Galileo's provenance records the TerraClimate
images and the LandScan year actually used. Per model: wall time, pixels per second, peak GPU memory.

usage: python -m fieldwise.collect.encode <collect_root> <paths.json> <out_dir> [device] [chunk]
paths.json: ModelPaths fields (fieldwise/fm/encode.py) pointing to the official code and weights you installed.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fieldwise.collect.reader import CollectionData
from fieldwise.fm import views
from fieldwise.fm.encode import Encoder, ModelPaths

SPECS = [('tessera', 'nano'), ('galileo', 'nano'), ('olmoearth', 'nano'), ('presto', 'default')]
ROOT, PATHS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
DEVICE = sys.argv[4] if len(sys.argv) > 4 else 'cuda'
CHUNK = int(sys.argv[5]) if len(sys.argv) > 5 else 4096


def save(path, write):
    tmp = path.with_name('.' + path.name + '.tmp')
    write(tmp)
    os.replace(tmp, path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = ModelPaths(**{k: Path(v) for k, v in json.loads(PATHS.read_text()).items()})
    d = CollectionData(ROOT)
    c = pd.read_parquet(ROOT / 'base_inputs/CASES.parquet').sort_values(['fid', 'stage']).reset_index(drop=True)
    prefix = c.case_id.str.rsplit(':', n=1).str[0]
    slots = np.stack([prefix + ':pre_sowing_calendar_year', prefix + ':tau1', c.case_id], 1)        # [cases, 3]
    keys = list(dict.fromkeys(slots.ravel()))
    own = set(c.case_id)
    means = np.full((len(c), 3, len(SPECS), 128), np.nan, np.float32)
    pix_rows = {}
    log = dict(cases=len(c), stages=c.stage.value_counts().to_dict(), windows=len(keys), models={}, galileo_provenance={})
    for j, (model, size) in enumerate(SPECS):
        torch.cuda.reset_peak_memory_stats() if DEVICE.startswith('cuda') else None
        t0 = time.time()
        enc = Encoder(model, size, paths, device=DEVICE, batch=CHUNK)
        t_load = time.time() - t0
        mean_of, n_enc = {}, 0
        for key in keys:
            fid, window = d.case(key)
            f = views.Field(d, fid, window)
            total, n = np.zeros(enc.dim, np.float64), 0
            rows = np.full((f.n, enc.dim), np.nan, np.float16) if (model == 'tessera' and key in own) else None
            for pos in f.chunks(CHUNK):
                v = views.presto(f, enc.pipeline, pos) if model == 'presto' else getattr(views, model)(f, pos)
                if model == 'galileo' and pos[0] == 0:
                    pv = v.get('provenance', {})
                    log['galileo_provenance'][key] = dict(tc=pv.get('tc'), landscan=pv.get('landscan'), viirs=pv.get('viirs'))
                emb = enc.encode(v).astype(np.float16).astype(np.float32)
                good = np.isfinite(emb).all(1)
                total += emb[good].sum(0, dtype=np.float64)
                n += int(good.sum())
                n_enc += len(pos)
                if rows is not None:
                    rows[pos] = emb.astype(np.float16)
            mean_of[key] = (total / n).astype(np.float32) if n else np.full(enc.dim, np.nan, np.float32)
            if rows is not None:
                pix_rows[key] = rows
        for i in range(len(c)):
            for s in range(3):
                means[i, s, j] = mean_of[slots[i, s]]
        secs = time.time() - t0
        log['models'][f'{model}_{size}'] = dict(
            weights_sha256=enc.weights_sha256, dim=enc.dim, load_seconds=round(t_load, 1), seconds=round(secs, 1),
            pixels_encoded=n_enc, pixels_per_second=round(n_enc / max(secs - t_load, 1e-9), 1),
            peak_gpu_mib=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if DEVICE.startswith('cuda') else None)
        del enc
        torch.cuda.empty_cache() if DEVICE.startswith('cuda') else None
        print(model, log['models'][f'{model}_{size}'], flush=True)
    order = list(c.case_id)
    sizes = [len(pix_rows[k]) for k in order]
    index = pd.DataFrame(dict(case_id=order, row_start=np.r_[0, np.cumsum(sizes)[:-1]], n_pixels=sizes))
    emb = np.concatenate([pix_rows[k] for k in order])
    save(OUT / 'tessera_pixels.npy', lambda p: np.save(open(p, 'wb'), emb))
    save(OUT / 'tessera_index.parquet', lambda p: index.to_parquet(p, index=False))
    save(OUT / 'fm_means.npz', lambda p: np.savez_compressed(open(p, 'wb'), case_ids=np.array(order), means=means,
                                                               windows=np.array(['pre_sowing_calendar_year', 'tau1', 'own']),
                                                               models=np.array([f'{m}_{s}' for m, s in SPECS])))
    log['tessera_rows'] = int(len(emb))
    log['tessera_nan_row_share'] = float((~np.isfinite(emb.astype(np.float32)).all(1)).mean())
    log['means_nan_share'] = float(np.isnan(means).any(-1).mean())
    save(OUT / 'ENCODE.json', lambda p: p.write_text(json.dumps(log, indent=1, default=str)))
    print(json.dumps({k: v for k, v in log.items() if k != 'galileo_provenance'}, indent=1, default=str))


if __name__ == '__main__':
    main()
