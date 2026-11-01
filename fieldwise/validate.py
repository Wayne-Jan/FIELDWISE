"""Checks of prepared FieldWise inputs (docs/INPUTS.md): structure, units and the timing rules every fit relies on.

Each check returns a table of issues (level: error or warning, check, detail, count). assert_valid() raises on errors.
Errors break an assumption of the method; warnings flag inputs that are legal but worth a look (for example earlier-year
cases that the time filters will leave out).
"""
import numpy as np
import pandas as pd

from .schedule import DAY_MS, DECISION_DELAY_DAYS
from .selection import PHASES, one_crop

ISSUE_COLUMNS = ['level', 'check', 'detail', 'count']
CASE_COLUMNS = ('case_id', 'year', 'field', 'stage', 'decision_ms', 'label_ms')


def _issues(rows):
    return pd.DataFrame([r for r in rows if r[3]], columns=ISSUE_COLUMNS)


def cases(C, phases=PHASES):
    """Case table: one row per field-season and decision stage.

    Required: case_id, year, field, stage, decision_ms, label_ms (UTC ms; label_ms may be missing for queries without
    labels). Optional: window_end_ms (end of the observation window), input_ms (time by which every input of the case is
    available), both not later than the decision; crop, which must hold a single crop (fit each crop separately).
    """
    missing = [c for c in CASE_COLUMNS if c not in C.columns]
    if missing:
        return _issues([('error', 'case_columns', f'missing columns {missing}', len(missing))])
    dec, lab = C.decision_ms.to_numpy(float), C.label_ms.to_numpy(float)
    rows = [('error', 'case_id_unique', 'duplicate case_id', int(C.case_id.duplicated().sum())),
            ('error', 'stage_known', f'stages outside {sorted(phases)}', int((~C.stage.isin(list(phases))).sum())),
            ('error', 'decision_finite', 'missing decision time', int((~np.isfinite(dec)).sum())),
            ('error', 'decision_before_label', 'labels available at or before the decision',
             int((np.isfinite(lab) & (lab <= dec)).sum())),
            ('error', 'one_case_per_stage', 'several cases of one field-season and stage',
             int(C.duplicated(['year', 'field', 'stage']).sum()))]
    for col, what in (('window_end_ms', 'window ends after the decision'), ('input_ms', 'inputs available after the decision')):
        if col in C.columns:
            rows.append(('error', f'{col}_by_decision', what, int((C[col].to_numpy(float) > dec).sum())))
    if 'crop' in C.columns:
        rows.append(('error', 'one_crop', 'several crops in one table (fit each crop separately)',
                     max(int(C.crop.nunique()) - 1, 0)))
    return _issues(rows)


def cutoffs(table, C, phases=PHASES, targets=None):
    """Cutoff table (year, phase, cutoff_ms) against the queries of every target year.

    A cutoff later than the earliest query decision of its year and phase would let a fit use information that is not
    available to that query (error); an earlier cutoff is legal but stricter than the manuscript rule (warning). Table and
    cases hold one crop.
    """
    crop = [t.crop for t in (table, C) if 'crop' in t.columns]
    n_crops = pd.concat(crop).nunique() if crop else 1
    if n_crops > 1:
        return _issues([('error', 'one_crop', 'several crops in the cutoff or case table (one crop per call)', n_crops - 1)])
    if table.duplicated(['year', 'phase']).any():
        return _issues([('error', 'cutoff_unique', 'several cutoffs for one year and phase',
                         int(table.duplicated(['year', 'phase']).sum()))])
    q = C.assign(phase=C.stage.map(phases))
    first = q.groupby(['year', 'phase']).decision_ms.min().rename('first_decision').reset_index()
    if targets is not None:
        first = first[first.year.isin(list(targets))]
    m = first.merge(table, on=['year', 'phase'], how='left')
    late = m.cutoff_ms > m.first_decision
    early = m.cutoff_ms < m.first_decision
    return _issues([('error', 'cutoff_present', 'target year and phase without a cutoff', int(m.cutoff_ms.isna().sum())),
                    ('error', 'cutoff_not_after_queries', 'cutoff later than the earliest query decision', int(late.sum())),
                    ('warning', 'cutoff_rule', 'cutoff earlier than the earliest query decision', int(early.sum()))])


def late_history(C, table, phases=PHASES):
    """Earlier-year cases that are not legal history for a target year because their decision or labels come after its
    cutoff (the fits leave them out; listed for information). Returns year, phase, decided_late, labelled_late."""
    one_crop(C)
    q = C.assign(phase=C.stage.map(phases))
    rows = []
    for (T, ph), cut in table.set_index(['year', 'phase']).cutoff_ms.items():
        h = q[(q.year < T) & (q.phase == ph)]
        rows.append(dict(year=T, phase=ph, history=len(h), decided_late=int((h.decision_ms > cut).sum()),
                         labelled_late=int((h.label_ms > cut).sum())))
    return pd.DataFrame(rows)


def bank(B, C):
    """Residual bank (bank.Bank) against the case table: pixel rows, finite predictions, labels where marked valid."""
    n_cases = len(B.cases)
    rows = [('error', 'bank_cases_known', 'bank cases not in the case table', int((~B.cases.case_id.isin(C.case_id)).sum())),
            ('error', 'bank_row_range', 'pixel rows outside the case range', int(((B.row < 0) | (B.row >= n_cases)).sum())),
            ('error', 'bank_rows_sorted', 'pixel rows not grouped by case', int((np.diff(B.row) < 0).sum())),
            ('error', 'bank_pred_finite', 'missing Base predictions on service pixels', int((~np.isfinite(B.pred)).sum())),
            ('error', 'bank_label_finite', 'pixels marked valid without a finite label',
             int((B.gt_valid & ~np.isfinite(B.target)).sum()))]
    n_gt = np.bincount(B.row[B.gt_valid].astype(np.int64), minlength=n_cases)
    rows.append(('warning', 'bank_cases_with_labels', 'cases without any labelled pixel (not used as history)',
                 int((n_gt == 0).sum())))
    return _issues(rows)


def daily(C, fs_row, n_days, fs_first_day, delay_days=DECISION_DELAY_DAYS):
    """Daily weather of every case ends with its observation window and is available at its decision: the last usable day
    (n_days rows from fs_first_day) must end by window_end_ms, and the end of that day plus the publication delay of the
    weather (delay_days, ERA5-Land 6 days) must not be later than decision_ms."""
    end = (np.asarray(fs_first_day)[np.asarray(fs_row)] + np.asarray(n_days)) * DAY_MS
    rows = []
    if 'window_end_ms' in C.columns:
        rows.append(('error', 'daily_within_window', 'usable daily weather beyond the window end',
                     int((end > C.window_end_ms.to_numpy(float)).sum())))
    rows.append(('error', 'daily_available_by_decision', 'usable daily weather published after the decision',
                 int((end + delay_days * DAY_MS > C.decision_ms.to_numpy(float)).sum())))
    return _issues(rows)


def interval_features(C, F, stage_index=None):
    """Retrieval descriptors of growing-degree-day intervals (columns containing '__stage<k>__', k = 0, 1, ...): at decision
    stage tau_j only intervals k < j have been reached and later ones must be missing."""
    stage_index = stage_index or {'tau1': 1, 'tau2': 2, 'tau3': 3, 'tau4': 4}
    j = C.stage.map(stage_index).to_numpy(float)
    bad = 0
    for k in range(8):
        cols = [c for c in F.columns if f'__stage{k}__' in c]
        if cols:
            reached = np.isfinite(F[cols].to_numpy(float)).any(1)
            bad += int((reached & np.isfinite(j) & (k >= j)).sum())
    return _issues([('error', 'interval_masked', 'descriptors of intervals not reached by the decision', bad)])


def embeddings(index, B):
    """Embedding index (case_id, row_start, n_pixels) against the bank: contiguous rows and one row per service pixel."""
    I = index.reset_index(drop=True)
    n_service = pd.Series(np.bincount(B.row.astype(np.int64), minlength=len(B.cases)), index=B.cases.case_id)
    known = I.case_id.isin(n_service.index)
    return _issues([('error', 'embedding_contiguous', 'row_start not contiguous',
                     int((np.diff(I.row_start) != I.n_pixels.iloc[:-1].to_numpy()).sum())),
                    ('error', 'embedding_cases', 'bank cases without embeddings', int((~B.cases.case_id.isin(I.case_id)).sum())),
                    ('error', 'embedding_pixels', 'embedding rows differ from service pixels',
                     int((I.n_pixels[known].to_numpy() != n_service.loc[I.case_id[known]].to_numpy()).sum()))])


def assert_valid(*issue_tables):
    """Raise ValueError listing every error; return the warnings."""
    t = pd.concat([x for x in issue_tables if len(x)], ignore_index=True) if any(len(x) for x in issue_tables) else \
        pd.DataFrame(columns=ISSUE_COLUMNS)
    err = t[t.level == 'error']
    if len(err):
        raise ValueError('invalid inputs:\n' + err.to_string(index=False))
    return t[t.level == 'warning']
