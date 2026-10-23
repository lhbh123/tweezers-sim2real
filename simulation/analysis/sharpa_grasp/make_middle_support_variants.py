"""Prepare paired physical support probes from one measured preheld state.

Both cases start from the same measured tool/arm/thumb/index state. Only the
middle finger's reset position and actuator target differ in the control case.
The probe adds closure to the saved target, so the schedule is relative to the
snapshot closure, not an absolute closure value. No simulation result is edited.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--retracted-pose', type=Path, required=True)
    parser.add_argument('--snapshot-closure', type=float, default=.75)
    parser.add_argument('--closures', type=float, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output file; existing experiment inputs are frozen.')
    snapshot = json.loads(args.snapshot.read_text())
    pose = json.loads(args.retracted_pose.read_text())
    names = snapshot['joint_names']
    middle = {name: value for name, value in pose['q_reference'].items()
              if name.startswith('right_middle_')}
    if len(middle) != 4 or any(name not in names for name in middle):
        parser.error('Expected all four W01 middle-finger joints in both inputs.')
    variants = []
    for closure in args.closures:
        if not 0 <= closure <= 1:
            parser.error('Requested absolute closures must be between 0 and 1.')
        relative = closure - args.snapshot_closure
        schedule = [[0., 0.], [1., 0.], [1.5, relative], [3.5, relative],
                    [4., 0.], [5., 0.]]
        for kind in ['support', 'retracted']:
            row = dict(name=f'{kind}_loose{closure:.2f}', closure_start=0.,
                       closure_end=0., closure_schedule=schedule)
            if kind == 'retracted':
                for field, state in [('initial_delta', 'q_actual'),
                                     ('target_delta', 'q_target')]:
                    row[field] = {name: value - snapshot[state][names.index(name)]
                                  for name, value in middle.items()}
            variants.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(variants, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'environments': len(variants),
                      'snapshot': str(args.snapshot),
                      'retracted_pose': str(args.retracted_pose)}))


if __name__ == '__main__':
    main()
