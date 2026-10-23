#!/usr/bin/env python3
"""90-observation broad-face-pinch BC initializer with strict task provenance.

Only successful complete episodes from explicit face-pinch/optional-support task
metadata are accepted. Offline imitation is not an RL or task-success claim.
"""
from pathlib import Path

from train_tripod_bc import fit, parser


def face_parser():
    p = parser()
    p.description = __doc__
    p.set_defaults(observation_size=90, action_size=6, task_kind='face-pinch')
    return p


def main():
    p = face_parser()
    args = p.parse_args()
    if args.observation_size != 90 or args.action_size != 6 or args.task_kind != 'face-pinch':
        p.error('This entrypoint only accepts native 90→6 face-pinch data; old 86-dimensional data cannot be mixed')
    args.entrypoint_file = Path(__file__).resolve()
    fit(args)


if __name__ == '__main__':
    main()
