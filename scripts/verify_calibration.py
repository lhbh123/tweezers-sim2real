#!/usr/bin/env python3
"""Recompute real validation errors into a new folder; never edit frozen evidence."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'calibration/src'))
from xhand_calib.solve import validate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    reports = {}
    for camera, result in [('wrist', 'handeye'), ('global', 'global_extrinsics')]:
        report = validate(ROOT / f'calibration/data/validation_{camera}',
                          ROOT / f'calibration/results/{camera}_intrinsics_v3.json',
                          ROOT / f'calibration/results/{result}.json',
                          overlay_dir=args.out / f'overlays_{camera}')
        (args.out / f'validation_{camera}.json').write_text(json.dumps(report, indent=2) + '\n')
        reports[camera] = dict(passed=report['passed'], statistics=report['statistics'])
    (args.out / 'summary.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports, indent=2))
    return 0 if all(r['passed'] for r in reports.values()) else 2


if __name__ == '__main__':
    sys.exit(main())
