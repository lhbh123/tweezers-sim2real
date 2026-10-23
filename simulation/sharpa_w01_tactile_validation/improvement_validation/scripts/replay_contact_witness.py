"""CPU diagnostic of raw and separation-offset contact witnesses.

Does not relabel the saved simulator acceptance result. No arbitrary projection
onto a desired face: candidate points are exactly p, p-s*n, p+s*n.
"""
import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run'); parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run = Path(args.run); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    report = json.loads((run/'report.json').read_text())
    z = np.load(run/'timeseries.npz'); times = z['time'].ravel()
    geo = report['face_definition']['geometry']; criteria = report['face_definition']['criteria']
    groups = defaultdict(list); frames = []
    with (run/'raw_contact_points.jsonl').open() as f:
        for line in f:
            row = json.loads(line); t = row['time_s']; env = row['env']
            frame = int(np.argmin(np.abs(times-t)))
            sums = np.zeros((3,2,2)); sums_all = np.zeros((2,2)); sums_validated = np.zeros((2,2))
            for point in row['points']['normal_contacts']:
                if point['tool'] not in ('left','right') or point['role'] not in ('thumb','index'): continue
                bi = ['left','right'].index(point['tool']); role = ['thumb','index'].index(point['role'])
                sign = -1 if bi == 0 else 1
                body = z['bodies'][frame, env, bi+1]
                normal_local = Rotation.from_quat(np.roll(body[3:7],-1)).inv().apply(point['normal_world'])
                raw = np.asarray(point['point_jaw_local_m'])
                step = point['separation_m'] * normal_local
                candidates = np.stack((raw, raw-step, raw+step))
                errors = np.abs(sign*candidates[:,0] - .5*geo['thickness_m'])
                zlocal = candidates[:,2]
                tip = zlocal > geo['length_m'] - geo['tip_length_m']
                width = np.where(tip, min(.005,geo['width_m']),geo['width_m'])
                z0 = np.where(tip,geo['length_m']-geo['tip_length_m'],0.)
                z1 = np.where(tip,geo['length_m'],geo['length_m']-geo['tip_length_m'])
                width_margin = width/2-np.abs(candidates[:,1])
                axial_margin = np.minimum(zlocal-z0,z1-zlocal)
                inward = point['inward_force_component_N']
                alignment = point['broad_face_inward_alignment_cos']
                qualified = (errors <= criteria['face_tolerance_m']) & (width_margin >= criteria['width_margin_m']) & (axial_margin >= criteria['axial_margin_m']) & (alignment >= criteria['min_inward_cos']) & (inward > 1e-8)
                if not criteria['allow_tip_contact']: qualified &= ~tip
                sums[:,bi,role] += qualified * inward
                sums_all[bi,role] += inward
                scalar = float(np.dot(point['force_on_tool_N'],point['normal_world']))
                witness_valid = (np.isfinite(candidates[1]).all() and scalar < -1e-8 and
                    abs(np.linalg.norm(normal_local)-1) <= 1e-3 and errors[1] <= 20e-6)
                sums_validated[bi,role] += inward * qualified[1 if witness_valid else 0]
                # A much stricter 20 micrometre witness residual is diagnostic
                # evidence only; the established 0.4 mm criterion is not enlarged.
                idx = int(np.argmin(errors))
                groups[point['tool']+'/'+point['role']].append({
                    'time_s': t, 'env': env, 'raw_error_m': float(errors[0]),
                    'minus_error_m': float(errors[1]), 'plus_error_m': float(errors[2]),
                    'min_error_m': float(errors[idx]), 'nearest_index': idx,
                    'separation_m': point['separation_m'], 'inward_N': inward,
                    'qualified': qualified.tolist(), 'normal_local_x': float(normal_local[0]),
                    'normal_force_scalar_N': scalar,
                    'raw_point_local_m': raw.tolist(), 'minus_point_local_m': candidates[1].tolist()})
            frames.append({'time_s': t,'env':env,'qualified_inward_N':sums.tolist(),
                           'validated_policy_inward_N':sums_validated.tolist(),'all_inward_N':sums_all.tolist()})
    summary = {}
    for key, rows in groups.items():
        a = {k:np.array([r[k] for r in rows]) for k in ('raw_error_m','minus_error_m','plus_error_m','min_error_m','nearest_index','inward_N','normal_force_scalar_N')}
        summary[key] = {'point_count':len(rows),
            'error_quantiles_um':{k:(np.quantile(a[k],[0,.5,.95,1])*1e6).tolist() for k in ('raw_error_m','minus_error_m','plus_error_m','min_error_m')},
            'nearest_counts_raw_minus_plus':[int((a['nearest_index']==i).sum()) for i in range(3)],
            'fraction_minus_residual_le_20um':float((a['minus_error_m']<=20e-6).mean()),
            'fraction_best_residual_le_20um':float((a['min_error_m']<=20e-6).mean()),
            'fraction_positive_scalar_force':float((a['normal_force_scalar_N']>0).mean())}
    result = {'run':str(run.resolve()),'raw_sha256':hashlib.sha256((run/'raw_contact_points.jsonl').read_bytes()).hexdigest(),
        'candidate_order':['raw','raw_minus_separation_times_normal','raw_plus_separation_times_normal'],
        'strict_witness_diagnostic_residual_m':20e-6,'original_face_criteria':criteria,
        'groups':summary,'frames':frames,
        'official_references':[
            'https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/_build/physx/latest/struct_px_contact_pair_point.html',
            'https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html'],
        'interpretation':'Official API defines negative separation as penetration and contact normal shape1-to-shape0, but does not promise returned contact point is on the sensor/tool surface. Candidate comparisons here are empirical geometric diagnostics, not a universal API contract.',
        'does_not_change_saved_acceptance':True}
    (out/(run.name+'_contact_witness_review.json')).write_text(json.dumps(result,indent=2))
    fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for key,rows in groups.items():
        raw=np.array([r['raw_error_m'] for r in rows])*1e3
        alt=np.array([r['minus_error_m'] for r in rows])*1e3
        axs[0].scatter(raw,alt,s=3,alpha=.4,label=key)
    axs[0].set_xlabel('Raw point outer-plane error (mm)');axs[0].set_ylabel('(p - separation * normal) error (mm)')
    axs[0].axhline(.02,color='k',ls='--',lw=.8,label='20 um diagnostic bound');axs[0].legend(fontsize=8)
    for bi,role,key in [(0,0,'left/thumb'),(1,1,'right/index')]:
        ts=np.array([r['time_s'] for r in frames if r['env']==0])
        raw=np.array([r['qualified_inward_N'][0][bi][role] for r in frames if r['env']==0])
        alt=np.array([r['validated_policy_inward_N'][bi][role] for r in frames if r['env']==0])
        axs[1].plot(ts,raw,label=key+' raw');axs[1].plot(ts,alt,'--',label=key+' validated tool witness')
    axs[1].set_xlabel('Simulation time (s), audited samples');axs[1].set_ylabel('Inward normal force passing unchanged face thresholds (N)');axs[1].legend(fontsize=8)
    for ax in axs:ax.grid(alpha=.2)
    fig.suptitle(run.name+' contact-coordinate diagnostic (does not overwrite acceptance)')
    fig.savefig(out/(run.name+'_contact_witness_review.png'),dpi=160);plt.close(fig)
    print(json.dumps({'run':run.name,'groups':summary},indent=2))


if __name__=='__main__':main()
