"""Frozen foundation-model encoders on fieldwise.fm.views inputs.

The model code and weights are not distributed with FieldWise; install each model from its official source under its own
licence and pass the locations (fieldwise/fm/README.md). Normalisation is the official one (TESSERA encode_pixels v1.1
statistics, Galileo Normalizer(std=True, std_multiplier=2) of normalization.json, Presto S1_S2_ERA5_SRTM.normalize,
OlmoEarth Normalizer COMPUTED). Eval mode, no gradients. Pixels without any observation get NaN rows.

    E = Encoder('olmoearth', 'nano', ModelPaths(olmoearth_code=..., olmoearth_model=...), device='cuda')
    for pos in F.chunks(): emb = E.encode(views.olmoearth(F, pos))        # [len(pos), 128] float32

Pooling: TESSERA its own pixel head; Galileo Encoder.average_tokens over unmasked tokens; Presto eval_task output; OlmoEarth
mean of the valid tokens of both modalities (fast_pass=False, patch_size=1; the official fast pass skips the attention mask).
"""
from __future__ import annotations

import hashlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# SHA-256 of the weights used for the FieldWise experiments (nano / default sizes); other versions are allowed with check=False
PINNED = {('tessera', 'nano'): 'bc9929e3643dfab2744c2ce14e7c14f698703a94e9562e4e0982f9d6540b0691',
          ('galileo', 'nano'): 'eab6acacf527886a508a555bc24efe4b4a2cf91b162e84cd1c4a1807e942180c',
          ('olmoearth', 'nano'): '2773fca48c238d78adde5e83b7d140a63d36c9e5f73746b8dbadaed743020378',
          ('presto', 'default'): '559c9d76f525bdf2dd3774da8b1e7f1510769b73baacac12e9e4d8d5c67fbe65'}
GALILEO_SHA = {'single_file_galileo.py': 'a94a2c1ee6ce541a2b09dcdaab6b5b3bd4edbee6b0108aa8a01bfe7a6395888d',
               'normalization.json': 'baea3828a8750e0b570cb2fad5cd30bb249f1f001988d842cc4246860e799ff3'}
DIMS = {'tessera': 128, 'presto': 128}


@dataclass
class ModelPaths:
    """Locations of the official model code and weights.

    tessera_code: folder of the TESSERA v2 student code (model.py, infer.py); tessera_weights: student_<size>.pt.
    galileo_code: folder with single_file_galileo.py and normalization.json; galileo_model: folder of <size>/encoder.pt.
    presto_code: the presto repository root; presto_weights: default_model.pt.
    olmoearth_code: folder holding the olmoearth_pretrain package; olmoearth_model: OlmoEarth-v1_2-<Size> folder.
    """
    tessera_code: Path | None = None
    tessera_weights: Path | None = None
    galileo_code: Path | None = None
    galileo_model: Path | None = None
    presto_code: Path | None = None
    presto_weights: Path | None = None
    olmoearth_code: Path | None = None
    olmoearth_model: Path | None = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def galileo_shift_div(code):
    """Official Normalizer(std=True, std_multiplier=2) with the integer-keyed normalization.json: (x - (mu - 2 sd)) / (4 sd) for
    S1, S2 (not NDVI), SRTM, ERA5 / TC / VIIRS, LandScan; NDVI 0/1, DW 0/1, WC 0/100, location 0/1; static arrays truncated as the
    official int64."""
    d = json.loads((Path(code) / 'normalization.json').read_text())

    def std(key, i):
        m, s = d[key]['mean'][i], d[key]['std'][i]
        return m - 2 * s, 4 * s
    out = {}
    for key, n, std_idx, div_fixed in (('13', 13, range(12), {}), ('16', 16, (0, 1), {i: 100. for i in range(11, 16)}),
                                       ('6', 6, range(6), {}), ('18', 18, (0,), {i: 100. for i in range(13, 18)})):
        shift, div = np.zeros(n), np.ones(n)
        for i, val in div_fixed.items():
            div[i] = val
        for i in std_idx:
            shift[i], div[i] = std(key, i)
        if n == 18:
            shift, div = np.trunc(shift), np.trunc(div)
        out[n] = (shift.astype(np.float32), div.astype(np.float32))
    return out


def presto_import_shims(presto_root):
    """Import Presto's model code without its Earth Engine / OpenMapFlow runtime dependencies (inference needs only the model,
    constants and checkpoint): minimal stub modules, then presto.presto is imported directly."""
    root = Path(presto_root) / 'presto'
    for name, path in (('presto', root), ('presto.dataops', root / 'dataops'), ('presto.dataops.pipelines', root / 'dataops' / 'pipelines')):
        m = types.ModuleType(name)
        m.__path__ = [str(path)]
        sys.modules[name] = m
    ee = types.ModuleType('ee')
    ee.Image = object
    ee.Geometry = types.SimpleNamespace(Polygon=object)
    ee.ImageCollection = ee.DateRange = ee.Date = lambda *a, **k: None
    sys.modules.setdefault('ee', ee)
    omf, exporter, engineer = (types.ModuleType(n) for n in ('openmapflow', 'openmapflow.ee_exporter', 'openmapflow.engineer'))
    exporter.create_ee_image = lambda *a, **k: None
    exporter.ee_safe_str = lambda v: v
    exporter.get_ee_task_list = lambda *a, **k: []
    engineer.calculate_ndvi = engineer.remove_bands = lambda x: x
    engineer.load_tif = lambda *a, **k: None
    for n, m in (('openmapflow', omf), ('openmapflow.ee_exporter', exporter), ('openmapflow.engineer', engineer)):
        sys.modules.setdefault(n, m)
    google, cloud, storage = (types.ModuleType(n) for n in ('google', 'google.cloud', 'google.cloud.storage'))
    storage.Client, storage.Blob = (lambda *a, **k: None), object
    google.cloud, cloud.storage = cloud, storage
    for n, m in (('google', google), ('google.cloud', cloud), ('google.cloud.storage', storage)):
        sys.modules.setdefault(n, m)


class Encoder:
    def __init__(self, model, size, paths, device='cpu', batch=1024, check=True):
        self.model, self.size, self.device, self.batch = model, size, torch.device(device), batch
        if model == 'tessera':
            sys.path.insert(0, str(paths.tessera_code))
            from model import load_model                                   # noqa: PLC0415
            from infer import encode_pixels                                # noqa: PLC0415
            self.weights = Path(paths.tessera_weights)
            self.net = load_model(str(self.weights), self.device).eval()
            self._encode_pixels, self.dim = encode_pixels, int(self.net.repr_dim)
        elif model == 'galileo':
            code = Path(paths.galileo_code)
            for name, digest in GALILEO_SHA.items():
                if check and sha(code / name) != digest:
                    raise ValueError(f'Galileo code differs from the pinned version: {name}')
            sys.path.insert(0, str(code))
            from single_file_galileo import Encoder as GE                  # noqa: PLC0415
            self.weights = Path(paths.galileo_model) / size / 'encoder.pt'
            self.net = GE.load_from_folder(self.weights.parent, device=self.device).to(self.device).eval()
            self.dim = int(json.loads((self.weights.parent / 'config.json').read_text())['model']['encoder']['embedding_size'])
            self.norm = {k: tuple(torch.as_tensor(a, device=self.device) for a in v) for k, v in galileo_shift_div(code).items()}
        elif model == 'presto':
            if size != 'default':
                raise ValueError('Presto size must be default')
            import importlib                                               # noqa: PLC0415
            presto_import_shims(paths.presto_code)
            self.weights = Path(paths.presto_weights)
            pm, um = importlib.import_module('presto.presto'), importlib.import_module('presto.utils')
            self.pipeline = importlib.import_module('presto.dataops.pipelines.s1_s2_era5_srtm')
            um.data_dir, um.default_model_path, um.device, pm.device = self.weights.parent, self.weights, self.device, self.device
            presto = pm.Presto.load_pretrained().to(self.device).eval()
            presto.requires_grad_(False)
            self.net, self.dim = presto.encoder, DIMS['presto']
        elif model == 'olmoearth':
            sys.path.insert(0, str(paths.olmoearth_code))
            from olmoearth_pretrain.model_loader import load_model_from_path        # noqa: PLC0415
            root = Path(paths.olmoearth_model)
            self.net = load_model_from_path(str(root)).to(self.device).eval()
            self.dim = json.loads((root / 'config.json').read_text())['model']['encoder_config']['embedding_size']
            self.weights = root / 'weights.pth'
        else:
            raise ValueError(model)
        self.weights_sha256 = sha(self.weights)
        pin = PINNED.get((model, size))
        if check and pin is not None and self.weights_sha256 != pin:
            raise ValueError(f'{model} {size} weights differ from the version used by FieldWise (pass check=False to allow)')
        (self.net.encoder if model == 'olmoearth' else self.net).requires_grad_(False)

    # ------------------------------------------------------------------------------------------------------------ encode
    def encode(self, v):
        assert v['model'] == self.model, (v['model'], self.model)
        n = len(v['positions'])
        out = np.full((n, self.dim), np.nan, np.float32)
        keep = np.flatnonzero(v['has_observation'])
        if not len(keep):
            return out
        out[keep] = getattr(self, f'_{self.model}')(v, keep)
        return out

    def _tessera(self, v, keep):
        kw = {}
        for name in ('asc', 'desc'):
            b = v[f's1_{name}_bands']
            kw[f's1_{name}_bands'] = b[keep] if b.shape[1] else None
            kw[f's1_{name}_doys'] = v[f's1_{name}_doys'] if b.shape[1] else None
        return self._encode_pixels(self.net, v['s2_bands'][keep], v['s2_doys'], s2_masks=v['s2_masks'][keep],
                                   device=self.device, batch_pixels=self.batch, **kw)

    def _galileo(self, v, keep):
        dev, N = self.device, self.norm
        T = len(v['months'])
        res = np.empty((len(keep), self.dim), np.float32)
        mo = torch.as_tensor(v['months'], device=dev)[None]
        f = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)   # noqa: E731
        for lo in range(0, len(keep), self.batch):
            ix = keep[lo:lo + self.batch]
            m = len(ix)
            with torch.inference_mode():
                o = self.net(s_t_x=((f(v['s_t_x'][ix]) - N[13][0]) / N[13][1])[:, None, None],
                             sp_x=((f(v['sp_x'][ix]) - N[16][0]) / N[16][1])[:, None, None],
                             t_x=(f(v['t_x'][ix]) - N[6][0]) / N[6][1], st_x=(f(v['st_x'][ix]) - N[18][0]) / N[18][1],
                             s_t_m=f(v['s_t_m'][ix])[:, None, None], sp_m=f(v['sp_m'][ix])[:, None, None],
                             t_m=f(v['t_m'][ix]), st_m=f(v['st_m'][ix]), months=mo.expand(m, T), patch_size=1)
                res[lo:lo + m] = self.net.average_tokens(*o[:8]).float().cpu().numpy()
        return res

    def _presto(self, v, keep):
        res = np.empty((len(keep), self.dim), np.float32)
        for lo in range(0, len(keep), self.batch):
            ix = keep[lo:lo + self.batch]
            kw = {k: torch.as_tensor(v[k][ix], device=self.device) for k in ('x', 'mask', 'dynamic_world', 'latlons')}
            with torch.inference_mode():
                res[lo:lo + len(ix)] = self.net(**kw, month=v['month'], eval_task=True).float().cpu().numpy()
        return res

    def _olmoearth(self, v, keep):
        from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality              # noqa: PLC0415
        from olmoearth_pretrain.data.normalize import Normalizer, Strategy               # noqa: PLC0415
        from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue        # noqa: PLC0415
        assert MISSING_VALUE == -99999
        norm = Normalizer(Strategy.COMPUTED)
        mods = (('sentinel2_l2a', Modality.SENTINEL2_L2A), ('sentinel1', Modality.SENTINEL1))
        x = {}
        fill = v.get('_test_filler')                                                    # tests only: value placed at missing tokens
        for name, mod in mods:
            raw = v[name]
            x[name] = np.where(raw == MISSING_VALUE, raw if fill is None else fill, norm.normalize(mod, raw)).astype(np.float32)
        ts = torch.as_tensor(v['timestamps'], device=self.device)[None]
        res = np.empty((len(keep), self.dim), np.float32)
        for lo in range(0, len(keep), self.batch):
            ix = keep[lo:lo + self.batch]
            kw = dict(timestamps=ts.expand(len(ix), -1, -1))
            for name, _ in mods:
                tok = v[f'{name}_token'][ix]
                kw[name] = torch.as_tensor(x[name][ix], device=self.device)
                kw[f'{name}_mask'] = torch.as_tensor(np.where(tok, MaskValue.ONLINE_ENCODER.value, MaskValue.MISSING.value).astype(np.float32),
                                                     device=self.device)
            with torch.inference_mode():
                tm = self.net.encoder(MaskedOlmoEarthSample(**kw), fast_pass=False, patch_size=1)['tokens_and_masks']
                acc, cnt = 0, 0
                for name, _ in mods:
                    t = getattr(tm, name)                                                  # [B, 1, 1, T, band sets, D]
                    mk = torch.as_tensor(v[f'{name}_token'][ix], device=self.device)[..., None]   # [B, 1, 1, T, 1, 1]
                    acc = acc + torch.where(mk, t, 0).sum(dim=(1, 2, 3, 4))
                    cnt = cnt + mk.expand_as(t[..., :1]).sum(dim=(1, 2, 3, 4))
                res[lo:lo + len(ix)] = (acc / cnt).float().cpu().numpy()
        return res
