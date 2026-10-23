"""Collect completed v15 evidence; fail before marking incomplete work delivered."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text())


def result(name, validated=False):
    data = read(name + '/evaluation.json')
    error = np.asarray(data['final_xy_error_m']) * 1000
    dist = np.linalg.norm(np.asarray(data['start'])[:, :2] - np.asarray(data['goal'])[:, :2], axis=1)
    item = dict(run=name, seed=data['seed'], successes=data['successes'], episodes=data['episodes'],
                checkpoint_sha256=data.get('checkpoint_sha256'),
                final_xy_error_mm=dict(mean=float(error.mean()), median=float(np.median(error)),
                                       p95=float(np.quantile(error, .95)), maximum=float(error.max())),
                transfer_distance_m=dict(minimum=float(dist.min()), maximum=float(dist.max())))
    if validated:
        val = read(name + '/validation.json')
        assert val['all_claims_consistent'] and val['archived_source_hashes_match']
        assert val['reconstructed_successes'] == data['successes']
        assert val['no_runtime_object_control_calls']
        assert val['tool_table_source_box_audit']['below_table_body_samples'] == 0
        usd = val['USD_audit']
        assert not usd['forbidden_tool_robot_or_block_joints']
        assert usd['all_objects_dynamic_with_gravity']
        assert usd['goal_markers_have_no_collision_or_rigid_body']
        if 'source_mesh_table_audit' in val:
            geo = val['source_mesh_table_audit']
            assert geo['below_table_link_samples'] == geo['joint_limit_violations'] == 0
        item['validation'] = {k: v for k, v in val.items() if k != 'rows'}
    return item


frozen = read('reports/pick_place15_frozen_candidate.json')
assert hashlib.sha256((ROOT / frozen['candidate']).read_bytes()).hexdigest() == frozen['sha256']
holdout = result('runs/pick_place15_ppo101_holdout20261017', True)
boundary = result('runs/pick_place15_ppo101_four_corners', True)
single = result('runs/pick_place15_ppo101_video_corner0', True)
for r in [holdout, boundary, single]:
    assert r['checkpoint_sha256'] == frozen['sha256']
geometry = read(single['run'] + '/substep_geometry_audit.json')
assert all(geometry['checks'].values())
deliveries = []
for name in ['four_corners', 'closeup']:
    receipt_path = f'reports/delivery_pick_place15_{name}.json'
    receipt = read(receipt_path)
    assert receipt['ok']
    video = f'outputs/ur5e_xhand_random_pick_place_{name}_v15.mp4'
    manifest = read(video.replace('.mp4', '.json'))
    sha = hashlib.sha256((ROOT / video).read_bytes()).hexdigest()
    assert sha == manifest.get('sha256', manifest.get('output_sha256'))
    deliveries.append(dict(video=video, sha256=sha, bytes=(ROOT / video).stat().st_size,
                           receipt=receipt_path, message_id=receipt['data']['message_id'],
                           native_render_manifest=manifest))

report = dict(
    status='completed_and_delivered', completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    task='UR5e + XHand preheld straight tweezers pick, transfer, place, release and withdraw',
    algorithm='Explicit geometric phase controller + goal-conditioned residual PPO; not end-to-end learned',
    workspace_xy_m=dict(lower=[.48, -.22], upper=[.76, .08], dimensions=[.28, .30]),
    randomization='Continuous uniform start proposal and independently sampled goal; resample goals until distance >= 0.12 m',
    checkpoint=frozen, training=read('reports/pick_place15_training_evidence.json'),
    development_comparison=[result('runs/' + name) for name in [
        'pick_place15_nominal51_outer70', 'pick_place15_ppo71_dev51',
        'pick_place15_ppo83_dev51', 'pick_place15_ppo101_dev51']],
    comparison_caveat='One shared development seed; candidate 101 also restores empty-tool closure to .8, so this is not an isolated PPO ablation.',
    independent_holdout=holdout, four_boundary_transfers=boundary, native_closeup_episode=single,
    every_physics_substep_geometry=geometry, deliveries=deliveries,
    limitations=[
        '64/64 is the result of this independent batch, not a guarantee of universal success.',
        'Validated tabletop workspace only; same cube size and already-held tool initial condition.',
        'Inputs are simulator object and goal states; no visual localization or real-robot validation.',
        'Random test transfer distances are >=12 cm; four separate corner cases cover the approximately 41 cm diagonal.',
        'Geometric motion phases and post-release grip restoration are explicit controller logic.',
        'Full substep self-intersection audit covers the single native episode; batch geometry is checked at control samples.',
        'Idealized passive spring tweezers, friction and source-vs-cooked collision approximations remain.'
    ])
(ROOT / 'reports/pick_place_revision_final.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
state = read('reports/pick_place_development_state.json')
state.update(status='completed_and_delivered', current_stage='independent validation and direct native video delivery complete',
             required_remaining=[], success_claimed=True, final_report='reports/pick_place_revision_final.json',
             selected_checkpoint=frozen['candidate'], holdout_successes=holdout['successes'],
             holdout_episodes=holdout['episodes'], deliveries=deliveries)
(ROOT / 'reports/pick_place_development_state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
print(json.dumps(dict(holdout=holdout['successes'], boundary=boundary['successes'], delivered=len(deliveries)), indent=2))
