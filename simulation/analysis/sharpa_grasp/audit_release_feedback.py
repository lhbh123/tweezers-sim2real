"""Check recorded force-clear and near-block withdrawal behavior independently."""
import argparse
import json
from pathlib import Path
import numpy as np


def longest(mask, dt):
    padded = np.r_[False, mask, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return float(np.max(edges[1::2]-edges[::2], initial=0)*dt)


def audit(run):
    r = json.loads((run/'evaluation.json').read_text())
    d = np.load(run/'trajectory.npz')
    cfg = r['task']['place_config']
    dt = float(d['dt'])
    requested = float(cfg.get('release_contact_clear_seconds', 0.))
    required = max(3, round(requested/dt))
    checks = []
    extras = d['release_extra_opening'] if 'release_extra_opening' in d.files else np.zeros(d['phase'].shape)
    for i in range(r['episodes']):
        active = d['active'][:, i].astype(bool)
        phase = d['phase'][:, i]
        forces = d['forces'][:, i].max(axis=-1)
        last = int(np.flatnonzero(active)[-1])
        release = active & (phase == 7)
        release_frames = np.flatnonzero(release)
        starts = np.flatnonzero(active & (phase == 8) & np.r_[True, phase[:-1] != 8])
        row = {'environment': i, 'recorded_success': bool(r['success_per_env'][i]),
               'max_extra_opening': float(extras[active, i].max()),
               'extra_starts_at_zero': bool(abs(extras[0, i]) < 1e-8),
               'extra_monotonic_and_bounded': bool(np.all(np.diff(extras[active, i]) >= -1e-7)
                                                  and extras[active, i].max() <= cfg.get('release_extra_opening_max', 0.)+1e-6)}
        row.update(final_phase=int(phase[last]), release_duration_control_samples_s=float(release.sum()*dt),
                   final_jaw_forces_N=d['forces'][last, i].tolist(),
                   final_tip_minus_block_m=(d['tip'][last, i]-d['block'][last, i, :3]).tolist(),
                   final_tool_retained=bool(d['retained'][last, i] < .035 and d['orientation_error'][last, i] < .3),
                   final_orientation_ok=bool(d['orientation_ok'][last, i]),
                   final_face_pinch=bool(d['face'][last, i]))
        if len(release_frames):
            # The last qualifying release measurement is saved with phase 8,
            # because the controller advances the phase after that measurement.
            release_measurement = release | (active & (phase == 8) & np.r_[False, phase[:-1] == 7])
            eligible = release_measurement & ((np.arange(len(phase))-release_frames[0])*dt >= 1.2-1e-8)
            cap = cfg.get('release_extra_opening_max', 0.)
            at_cap = release & (extras[:, i] >= cap-1e-6) if cap > 0 else np.zeros_like(release)
            tail = release_frames[-max(1, round(1./dt)):]
            row.update(longest_eligible_release_clear_s=longest(eligible & (forces < .003), dt),
                       release_at_extra_opening_cap_s=float(at_cap.sum()*dt),
                       last_release_second_max_jaw_force_N=float(forces[tail].max()),
                       last_release_second_mean_jaw_force_N=float(forces[tail].mean()))
        if not len(starts):
            row.update(withdrawal_reached=False)
            if phase[last] == 7 and release.sum()*dt > 12.:
                row['failure_explanation'] = 'Release phase exceeded 12 s with residual jaw contact; no withdrawal was commanded.'
            checks.append(row)
            continue
        k = int(starts[0])
        j = k
        while j >= 0 and (j == k or phase[j] == 7) and forces[j] < .003:
            j -= 1
        row.update(withdrawal_reached=True, withdrawal_entry_time_s=(k+1)*dt,
                   jaw_forces_at_entry_N=d['forces'][k, i].tolist(),
                   force_clear_control_samples_before_entry=k-j,
                   force_clear_duration_s=(k-j)*dt,
                   force_clear_gate_satisfied=bool(requested <= 0 or k-j >= required),
                   max_force_in_required_clear_window_N=float(forces[max(0, k-required+1):k+1].max()),
                   extra_opening_at_entry=float(extras[k, i]))
        closure = r['task']['closure_bias'] + d['applied'][:, i, 3]*r['task']['closure_scale']
        above = d['tip'][:, i, 2] - d['block'][:, i, 2]
        # The command at k is chosen from measurements at k-1. The phase
        # recorded at k includes any transition after the simulation step.
        near = np.zeros(len(phase), dtype=bool)
        near[1:] = active[1:] & (phase[:-1] == 8) & (above[:-1] < .020)
        speed = np.linalg.norm(d['applied'][:, i, :3], axis=-1)*r['task']['max_wrist_step']/dt
        if near.any():
            expected = cfg['open_closure'] - extras[:, i]
            row.update(near_block_withdraw_control_samples=int(near.sum()),
                       near_block_max_command_speed_m_s=float(speed[near].max()),
                       near_block_max_closure_baseline_error=float(np.abs(closure[near]-expected[near]).max()),
                       release_opening_preserved_until_clear=bool(np.all(np.abs(closure[near]-expected[near]) < 1e-6)),
                       initial_withdraw_speed_limit_satisfied=bool(cfg.get('withdraw_initial_speed_m_s') is None
                           or speed[near].max() <= cfg['withdraw_initial_speed_m_s']+1e-6))
            cap = cfg.get('withdraw_initial_speed_m_s')
            violations = np.flatnonzero(near & (speed > cap+1e-6)) if cap is not None else []
            row['near_block_speed_limit_exceedances'] = [
                {'time_s': float((j+1)*dt), 'command_speed_m_s': float(speed[j]),
                 'prior_tip_clearance_m': float(above[j-1]),
                 'prior_command_speed_m_s': float(speed[j-1]),
                 'jaw_forces_N': d['forces'][j, i].tolist()} for j in violations]
        checks.append(row)
    properties = ['extra_starts_at_zero', 'extra_monotonic_and_bounded', 'force_clear_gate_satisfied',
                  'release_opening_preserved_until_clear', 'initial_withdraw_speed_limit_satisfied']
    return {'run': str(run), 'requested_clear_duration_s': requested,
            'configured_control_parameters': {name: cfg.get(name) for name in
                ['open_closure','closed_closure','retreat_closure','release_extra_opening_max',
                 'release_extra_opening_rate','release_contact_clear_seconds','withdraw_initial_speed_m_s']},
            'all_observed_feedback_checks_passed': all(row.get(name, True) for row in checks for name in properties),
            'withdrawal_reached_count': sum(row['withdrawal_reached'] for row in checks),
            'rows': checks,
            'scope': 'Active saved control frames. Block-jaw force comes from the contact sensor at the final physics substep of each control interval. This audit verifies consecutive control samples, not a bound on every physics-substep force.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    args = p.parse_args()
    result = audit(args.run)
    text = json.dumps(result, indent=2)+'\n'
    (args.run/'independent_release_feedback_audit.json').write_text(text)
    print(text)
