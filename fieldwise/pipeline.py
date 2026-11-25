"""Direct entry point for running FieldWise on any pixel-level dataset.

Bring a pixel table (one row per pixel with case_id, year, field, stage, pred, target) and a matching pixel feature or
embedding matrix [N_pixels, D]. Everything runs through the exact same engines as the manuscript (retrieval, RPT,
calibration, forward strength selection and the optional decision layer) with strict temporal legality: target year T
only ever uses labelled reference cases from earlier years.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from . import bank as fb, calibration, decision, retrieval, rpt, selection, strength

REQUIRED_COLUMNS = ('case_id', 'year', 'field', 'stage', 'pred', 'target')
_YEAR_MS = 365 * 86_400_000


@dataclass
class PixelDataset:
    """Prepared FieldWise inputs built from a flat pixel-level table."""

    bank: fb.Bank
    emb: np.ndarray
    emb_index: pd.DataFrame
    features: pd.DataFrame
    groups: dict
    geometry: dict | None
    order: np.ndarray


@dataclass
class FieldWiseResult:
    """Outputs of correct_pixels(): pixel-level predictions, case summary, strengths and intermediate objects."""

    pixels: pd.DataFrame
    cases: pd.DataFrame
    choices: dict
    bank: fb.Bank
    references: dict
    s15: np.ndarray


def _default_timestamps(years):
    """Strict year-level timestamps (decision mid-year, label end-of-year) when explicit timestamps are omitted."""
    y = np.asarray(years, np.int64) - 1970
    decision_ms = y * _YEAR_MS + 180 * 86_400_000
    label_ms = y * _YEAR_MS + 330 * 86_400_000
    return decision_ms, label_ms


def _neighbours_xy(x, y, radius=15.0):
    """CSR neighbour lists within radius metres for projected coordinates (x, y)."""
    xy = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    lst = cKDTree(xy).query_ball_point(xy, r=radius + 0.5)
    lens = np.fromiter((len(v) for v in lst), np.int64, len(lst))
    return np.concatenate([[0], np.cumsum(lens)]), np.concatenate([np.sort(v) for v in lst]).astype(np.int64)


def from_pixels(pixels, emb, case_features=None, groups=None, radius=15.0):
    """Pack a flat pixel-level DataFrame and pixel embeddings [N_pixels, D] into a FieldWise PixelDataset.

    pixels columns:
      required: case_id, year, field, stage, pred, target (target may be NaN for unlabelled target-year pixels)
      optional: gt_valid (defaults to isfinite(target)), decision_ms, label_ms, lon/lat (WGS 84) or x/y (metres).
    emb: [len(pixels), D] array of pixel embeddings or pixel-level remote-sensing features (aligned with pixels rows).
    case_features: optional DataFrame indexed by or containing case_id for reference-field retrieval; if omitted,
      the field-mean of emb per case is used automatically.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in pixels.columns]
    if missing:
        raise ValueError(f'pixels is missing required columns: {missing}')
    E = np.asarray(emb)
    if E.ndim != 2 or len(E) != len(pixels):
        raise ValueError(f'emb must have shape ({len(pixels)}, D), got {E.shape}')

    # Sort stably by (year, stage, case_id) so each case occupies one contiguous pixel slice.
    sort_df = pixels[['year', 'stage', 'case_id']].copy()
    sort_df['_ord'] = np.arange(len(pixels))
    order = sort_df.sort_values(['year', 'stage', 'case_id', '_ord'], kind='mergesort')['_ord'].to_numpy()
    df = pixels.iloc[order].reset_index(drop=True)
    E = np.ascontiguousarray(E[order], dtype=np.float16)

    cid_codes, unique_cids = pd.factorize(df['case_id'], sort=False)
    counts = np.bincount(cid_codes, minlength=len(unique_cids)).astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    first = starts

    cases = pd.DataFrame({
        'case_id': unique_cids.astype(object),
        'year': df['year'].to_numpy()[first].astype(np.int64),
        'field': df['field'].to_numpy()[first],
        'stage': df['stage'].astype(str).to_numpy()[first],
        'pix0': starts,
        'n_service': counts,
    })
    if 'decision_ms' in df.columns and 'label_ms' in df.columns:
        cases['decision_ms'] = df['decision_ms'].to_numpy(np.int64)[first]
        cases['label_ms'] = df['label_ms'].to_numpy(np.int64)[first]
    else:
        dms, lms = _default_timestamps(cases['year'].to_numpy())
        cases['decision_ms'] = df['decision_ms'].to_numpy(np.int64)[first] if 'decision_ms' in df.columns else dms
        cases['label_ms'] = df['label_ms'].to_numpy(np.int64)[first] if 'label_ms' in df.columns else lms

    target = df['target'].to_numpy(np.float32)
    gt_valid = df['gt_valid'].to_numpy(bool) if 'gt_valid' in df.columns else np.isfinite(target)
    gt_valid = gt_valid & np.isfinite(target)
    pred = df['pred'].to_numpy(np.float32)
    pix_id = np.arange(len(df), dtype=np.int64)

    bank = fb.Bank(cases=cases, row=cid_codes.astype(np.int64), pixel=pix_id, pred=pred, target=target, gt_valid=gt_valid)
    M = bank.moments()
    bank.cases['rbar'] = np.where(M['n'].to_numpy() > 0, M['rbar'].to_numpy(), np.nan)

    emb_index = pd.DataFrame({'case_id': cases['case_id'], 'row_start': starts, 'n_pixels': counts})

    # Spatial 15 m smoothing neighbourhoods if pixel coordinates are provided.
    geometry = None
    has_ll = 'lon' in df.columns and 'lat' in df.columns
    has_xy = 'x' in df.columns and 'y' in df.columns
    if radius and (has_ll or has_xy):
        geometry = {}
        c0 = df['lon' if has_ll else 'x'].to_numpy(np.float64)
        c1 = df['lat' if has_ll else 'y'].to_numpy(np.float64)
        for cid, s, n in zip(cases['case_id'], starts, counts):
            sl = slice(int(s), int(s + n))
            geometry[cid] = rpt.neighbours(c0[sl], c1[sl], radius=radius) if has_ll else _neighbours_xy(c0[sl], c1[sl], radius)

    # Retrieval summary table: use caller-provided features or default to field-mean embeddings.
    if case_features is None:
        E32 = np.where(np.isfinite(E), E.astype(np.float64), 0.0)
        fin = np.isfinite(E).all(1).astype(np.float64)
        den = np.bincount(cid_codes, weights=fin, minlength=len(cases))
        means = np.zeros((len(cases), E.shape[1]), dtype=np.float64)
        for j in range(E.shape[1]):
            num = np.bincount(cid_codes, weights=E32[:, j] * fin, minlength=len(cases))
            means[:, j] = np.divide(num, den, out=np.full(len(cases), np.nan), where=den > 0)
        cols = [f'emb_mean_{j}' for j in range(E.shape[1])]
        feat = pd.DataFrame(means, columns=cols)
        feat.insert(0, 'case_id', cases['case_id'].to_numpy())
        groups = {'emb': cols}
    else:
        feat = case_features.copy()
        if 'case_id' not in feat.columns:
            feat = feat.reset_index().rename(columns={feat.index.name or 'index': 'case_id'})
        feat = feat.set_index('case_id').loc[cases['case_id']].reset_index()
        if groups is None:
            num_cols = [c for c in feat.columns if c != 'case_id' and pd.api.types.is_numeric_dtype(feat[c])]
            groups = {'features': num_cols}

    return PixelDataset(bank=bank, emb=E, emb_index=emb_index, features=feat, groups=groups, geometry=geometry, order=order)


def correct_pixels(
    pixels,
    emb,
    case_features=None,
    groups=None,
    *,
    k=50,
    radius=15.0,
    G=None,
    rhat=None,
    use_calibration=True,
    device=None,
    seed=0,
):
    """Run strictly causal rolling FieldWise correction directly on pixel-level data.

    For each target year after the earliest year, learns RPT (within-field transfer), linear calibration (and optional
    environment offset G + decision readout rhat) strictly from earlier labelled years, selects component strengths by
    forward selection on historical years, and returns pixel-level Base, Correction-only and FieldWise predictions
    aligned with the input pixels row order.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = from_pixels(pixels, emb, case_features=case_features, groups=groups, radius=radius)
    bank = ds.bank
    C = bank.cases.copy()
    M = bank.moments()
    C = pd.concat([C, M[['n', 'sy', 'sp', 'syy', 'syp', 'spp', 'sr', 'srr', 'p_mean', 'p_sd', 'log_n', 'mse']]], axis=1)
    C['w'] = selection.field_season_weights(C)
    cut = selection.cutoffs(C)

    # 1. Reference-field retrieval (strictly earlier years with available labels)
    refs = retrieval.select_references(C[['case_id', 'year', 'stage', 'decision_ms', 'label_ms']], ds.features, ds.groups, k=k)

    # 2. Within-field Residual Pattern Transfer (RPT)
    engine = rpt.RPT(bank, ds.emb, ds.emb_index, refs, geometry=ds.geometry, device=device, seed=seed)
    s15 = np.full(len(bank.pred), np.nan, dtype=np.float32)
    b = bank.bounds
    years = sorted(int(y) for y in C['year'].unique())
    for T in years[1:]:
        _, _, _, pix_T = rpt.run_year(engine, T, cutoff=cut)
        for q, v in pix_T.items():
            s15[b[q]:b[q + 1]] = v
    s15_fill = np.nan_to_num(s15, nan=0.0).astype(np.float64)

    # Per-case RPT moments on labelled pixels
    row = bank.row.astype(np.int64)
    gt = bank.gt_valid
    r_pix = bank.target.astype(np.float64) - bank.pred.astype(np.float64)
    nc = len(C)
    C['sS'] = np.bincount(row[gt], weights=s15_fill[gt], minlength=nc)
    C['srS'] = np.bincount(row[gt], weights=r_pix[gt] * s15_fill[gt], minlength=nc)
    C['sSS'] = np.bincount(row[gt], weights=s15_fill[gt] ** 2, minlength=nc)

    # 3. Field-wide components (Calibration + optional G)
    labelled = C['n'].to_numpy() > 0
    if use_calibration and labelled.any():
        coef = calibration.fit_affine(C[labelled], min_years=2, cutoff=cut, targets=years)
        Cal = calibration.calibration_offset(coef, C['year'], C['stage'], C['p_mean'])
    else:
        Cal = np.full(nc, np.nan)
    G_arr = np.full(nc, np.nan) if G is None else pd.Series(G).reindex(C['case_id']).to_numpy(np.float64)

    # 4. Forward selection of strengths (gamma, beta, kappa) using earlier labelled years
    m_all = C[list(strength.MOMENTS)].to_numpy(np.float64)
    m_lab = m_all[labelled]
    picks, _ = strength.select_strengths(C[labelled].reset_index(drop=True), m_lab, G_arr[labelled], Cal[labelled],
                                         cutoff=cut, targets=years)
    gamma = np.array([picks[int(y)][0] for y in C['year']], dtype=np.float64)
    beta = np.array([picks[int(y)][1] for y in C['year']], dtype=np.float64)
    kappa = np.array([picks[int(y)][2] for y in C['year']], dtype=np.float64)
    c_off = gamma * np.nan_to_num(G_arr) + kappa * np.nan_to_num(Cal)

    # 5. Optional decision layer (if readout predictions rhat are supplied)
    if rhat is not None:
        rhat_arr = pd.Series(rhat).reindex(C['case_id']).to_numpy(np.float64)
        U, c_off = decision.offsets(rhat_arr, G_arr, Cal, gamma, kappa)
        alphas, _ = decision.select_alpha(C[labelled].reset_index(drop=True), m_lab, U[labelled], beta[labelled],
                                          cutoff=cut, targets=years)
        a_col = np.array([int(np.flatnonzero(np.isclose(decision.ALPHAS, alphas[int(y)]))[0]) for y in C['year']])
        u_off = U[np.arange(nc), a_col]
        alpha_case = np.array([alphas[int(y)] for y in C['year']], dtype=np.float64)
    else:
        alphas = {y: None for y in years}
        u_off = c_off.copy()
        alpha_case = np.full(nc, np.nan)

    # Zero out corrections on the earliest year (no historical bank)
    first_y = years[0]
    is_first = C['year'].to_numpy() == first_y
    beta[is_first] = 0.0
    c_off[is_first] = 0.0
    u_off[is_first] = 0.0

    # Assemble pixel-level outputs in the caller's original row order
    inv = np.empty_like(ds.order)
    inv[ds.order] = np.arange(len(ds.order))
    b_pix = beta[row]
    c_pix = c_off[row]
    u_pix = u_off[row]
    base_pix = bank.pred.astype(np.float64)
    rpt_pix = b_pix * s15_fill
    corr_only = base_pix + rpt_pix + c_pix
    fw_pred = base_pix + rpt_pix + u_pix

    out_px = pixels.copy()
    out_px['pred_base'] = base_pix[inv]
    out_px['s15'] = s15[inv]
    out_px['within_field'] = rpt_pix[inv]
    out_px['field_offset_corr_only'] = c_pix[inv]
    out_px['field_offset'] = u_pix[inv]
    out_px['pred_correction_only'] = corr_only[inv]
    out_px['pred_fieldwise'] = fw_pred[inv]

    C['G'] = G_arr
    C['Cal'] = Cal
    C['gamma'] = gamma
    C['beta'] = beta
    C['kappa'] = kappa
    C['alpha'] = alpha_case
    C['field_offset_corr_only'] = c_off
    C['field_offset'] = u_off
    choices = {int(y): dict(gamma=picks[y][0], beta=picks[y][1], kappa=picks[y][2], alpha=alphas[y]) for y in years}
    return FieldWiseResult(pixels=out_px, cases=C, choices=choices, bank=bank, references=refs, s15=s15[inv])
