"""Real-record consistency and adversarial mutation checks, without simulation.

Every mutation is made to an in-memory copy. Original trajectory/task/USD files
are hashed before/after. A rejection can be a false check or a schema exception.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'analysis'), str(ROOT / 'src')]
from audit_recorded_tripod_support import audit_support
from kinematics import Kinematics
spec = importlib.util.spec_from_file_location('independent_rl_validator', ROOT / 'scripts/validate_rl_run.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(run):
    rows = json.loads((run / 'trajectory.json').read_text())
    task = json.loads((run / 'task.json').read_text())
    usd = validator.usd_audit(run / 'final_scene.usda')
    return rows, task, usd


def evaluate(rows, task, usd, kin):
    try:
        ok, checks, evidence = audit_support(rows, task, usd, np.ones(len(rows), bool), kin,
                                             validator.rotations_from_wxyz)
        return dict(accepted=all(checks.values()), failed_checks=[k for k, v in checks.items() if not v],
                    support_gate_fraction=float(ok.mean()), evidence=evidence)
    except (ValueError, KeyError, TypeError, IndexError) as error:
        return dict(accepted=False, exception_type=type(error).__name__, exception=str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--legacy-run', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'analysis/tripod_support_audit_tests.json')
    args = parser.parse_args()
    paths = [args.run / f for f in ['trajectory.json', 'task.json', 'final_scene.usda']]
    if args.legacy_run:
        paths += [args.legacy_run / f for f in ['trajectory.json', 'task.json', 'final_scene.usda']]
    before = {str(p): sha(p) for p in paths}
    rows, task, usd = load(args.run)
    kin = Kinematics()
    baseline = evaluate(rows, task, usd, kin)
    if not baseline['accepted']:
        raise RuntimeError('Unmodified real record failed: ' + str(baseline))
    # Real first row contains thumb/index/middle and an actual tail-web contact.
    row_index = next(i for i, r in enumerate(rows) if any(
        p['within_web_region'] for p in r['grasp_contact_substeps'][0]['grasp_contact_points']['normal_contacts']))
    mutations = []
    def add(name, change):
        selected = [copy.deepcopy(rows[row_index])]
        t = copy.deepcopy(task)
        change(selected[0], t)
        result = evaluate(selected, t, usd, kin)
        result.pop('evidence', None)
        mutations.append(dict(name=name, correctly_rejected=not result['accepted'], result=result))
    def first_point(row):
        return row['grasp_contact_substeps'][0]['grasp_contact_points']['normal_contacts'][0]
    add('raw_role_metadata_changed', lambda r,t: first_point(r).__setitem__('role', 'thumb'))
    add('raw_normal_force_changed', lambda r,t: first_point(r)['force_on_tool_N'].__setitem__(2, 1.0))
    add('raw_web_ROI_membership_flag_changed', lambda r,t: first_point(r).__setitem__('within_web_region', False))
    add('anatomical_ROI_center_changed', lambda r,t: t['web_contact_region']['center_m'].__setitem__(0, 1.0))
    add('tail_contact_relabelled_as_blade', lambda r,t: first_point(r).__setitem__('tool', 'left'))
    add('unknown_robot_body_with_valid_looking_role', lambda r,t: first_point(r).__setitem__('robot', 'fake_thumb_link'))
    add('logged_support_pass_flag_changed', lambda r,t: r.__setitem__('grasp_support_acceptable', not r['grasp_support_acceptable']))
    add('one_physics_substep_omitted', lambda r,t: r['grasp_contact_substeps'].pop(0))
    add('one_physics_substep_duplicated', lambda r,t: r['grasp_contact_substeps'].__setitem__(1, copy.deepcopy(r['grasp_contact_substeps'][0])))
    add('mean_metadata_downgraded_to_legacy', lambda r,t: t.pop('grasp_support_temporal_aggregation'))
    add('all_substeps_omitted', lambda r,t: r.pop('grasp_contact_substeps'))
    add('last_endpoint_state_changed', lambda r,t: r['tool'].__setitem__(0, r['tool'][0] + .001))
    add('substep_joint_pose_changed', lambda r,t: r['grasp_contact_substeps'][0]['joint_pos'].__setitem__(0, r['grasp_contact_substeps'][0]['joint_pos'][0] + .02))
    add('logged_mean_force_changed', lambda r,t: r['grasp_support']['web_normal_force_w'].__setitem__(2, .5))
    legacy = None
    if args.legacy_run:
        lr, lt, lu = load(args.legacy_run)
        if lt.get('grasp_support_temporal_aggregation') or any('grasp_contact_substeps' in r for r in lr):
            raise ValueError('Legacy test must use an actual instantaneous record')
        legacy = evaluate(lr, lt, lu, kin)
    after = {str(p): sha(p) for p in paths}
    report = dict(run=str(args.run), baseline=baseline, mutations=mutations,
                  legacy_run=str(args.legacy_run) if args.legacy_run else None, legacy=legacy,
                  input_sha256=before, original_records_unchanged=before == after,
                  auditor_sha256=sha(ROOT / 'analysis/audit_recorded_tripod_support.py'),
                  all_tests_passed=baseline['accepted'] and all(m['correctly_rejected'] for m in mutations)
                                   and before == after and (legacy is None or legacy['accepted']))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['all_tests_passed', 'original_records_unchanged']}, indent=2))
    print('Baseline samples:', baseline['evidence']['physics_samples_reconstructed'])
    print('Mutation rejections:', sum(m['correctly_rejected'] for m in mutations), '/', len(mutations))
    if legacy:
        print('Legacy accepted:', legacy['accepted'])
    print('Report:', args.output)
    return 0 if report['all_tests_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
