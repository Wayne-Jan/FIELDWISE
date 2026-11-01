"""Collection steps of one season (python -m fieldwise.collect <step> --root <season>/collect --config <collection.json>).

Start the workspace with python -m fieldwise.collect.init_fields; --plan prints a step's recipe without downloading.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

STEPS = ('base', 'soil', 'terrain', 'build-base', 'lst', 'build-environment', 'fm', 'rtc-pre', 'rtc')


def commands(root, cfg):
    r, f, b = root / 'raw', root / 'fields', root / 'base_inputs'
    return {
        'base': ('extract_base_inputs', [f, r / 'base_inputs']),
        'soil': ('soil_official', [f, r / 'base_inputs/soil_official']),
        'terrain': ('build_terrain', [r / 'base_inputs', f, root / 'terrain']),
        'build-base': ('build_base_inputs', [f, r / 'base_inputs', root / 'terrain', b]),
        'lst': ('extract_lst', [f, r / 'lst']),
        'build-environment': ('build_g_inputs', [b, r / 'base_inputs', f, r / 'lst', root / 'g_inputs']),
        'fm': ('extract_fm_products', [f, b, r / 'fm_products']),
        'rtc-pre': ('extract_s1_rtc', [f, b, r / 's1_rtc_pre', cfg['pre_start'], cfg['base_start']]),
        'rtc': ('extract_s1_rtc', [f, b, r / 's1_rtc', cfg['base_start'], cfg['end']]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('step', choices=STEPS)
    ap.add_argument('--root', type=Path, required=True, help='Season workspace written by fieldwise.collect.init_fields')
    ap.add_argument('--config', type=Path, required=True, help='<root>/collection.json')
    ap.add_argument('--plan', action='store_true')
    a = ap.parse_args()
    os.environ['FIELDWISE_COLLECTION_CONFIG'] = str(a.config.resolve())
    from .settings import config
    cfg = config()
    recipe = commands(a.root, cfg)
    if a.plan:
        print(json.dumps(dict(step=a.step, config=cfg, command=[str(x) for x in recipe[a.step][1]],
                              network=a.step in ('base', 'soil', 'lst', 'fm', 'rtc-pre', 'rtc')), indent=2))
        return
    # Never reuse caches with another grid, date interval or field population.
    fields = a.root / 'fields/PIXELS.parquet'
    if not fields.exists():
        ap.error('Initialise the fields first: python -m fieldwise.collect.init_fields')
    identity = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode() + fields.read_bytes()).hexdigest()
    receipt = a.root / 'COLLECTION_ID.json'
    if receipt.exists() and json.loads(receipt.read_text())['sha256'] != identity:
        ap.error('Configuration or fields changed: use a new acquisition workspace')
    receipt.write_text(json.dumps(dict(sha256=identity, config=cfg), indent=2))
    module, args = recipe[a.step]
    sys.argv = [module, *map(str, args)]
    runpy.run_module('fieldwise.collect.' + module, run_name='__main__')


if __name__ == '__main__':
    main()
