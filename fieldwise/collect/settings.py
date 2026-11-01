"""Explicit acquisition configuration; credentials are never written to outputs."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def config():
    p = os.environ.get('FIELDWISE_COLLECTION_CONFIG')
    if not p:
        raise ValueError('Run python -m fieldwise.collect with --config')
    d = json.loads(Path(p).read_text())
    if d.get('crop') != 'corn':
        raise ValueError('This collection recipe currently prepares maize only')
    for key in ('pre_start', 'weather_start', 'base_start', 'fm_end', 'end'):
        pd.Timestamp(d[key])
    if not (d['pre_start'] < d['weather_start'] <= d['base_start'] < d['fm_end'] <= d['end']):
        raise ValueError('Expected pre_start < weather_start <= base_start < fm_end <= end')
    return d


def project():
    p = os.environ.get('FIELDWISE_EE_PROJECT')
    if not p:
        raise ValueError('Set FIELDWISE_EE_PROJECT to your own Earth Engine project')
    return p


def weather_cells(fields):
    """Union of the four ERA5-Land cell centres surrounding each field centre."""
    f = pd.read_parquet(Path(fields) / 'FIELDS.parquet')
    cells = set()
    for lon, lat in f[['lon', 'lat']].to_numpy():
        a, b = np.floor(lon * 10) / 10, np.floor(lat * 10) / 10
        cells.update((round(a + dx, 1), round(b + dy, 1)) for dx in (0, .1) for dy in (0, .1))
    return [list(c) for c in sorted(cells)]


def month_edges(start, end):
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    return pd.DatetimeIndex([a, *pd.date_range(a, b, freq='MS').tolist(), b]).unique().sort_values()
