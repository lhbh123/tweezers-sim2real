"""Describe saved W01 block motion independently of the recorded success flag."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def longest(mask,dt):
    count=best=0
    for value in mask:
        count=count+1 if value else 0
        best=max(best,count)
    return best*dt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    with np.load(a.run/'trajectory.npz',allow_pickle=False) as archive:
        d={k:archive[k] for k in archive.files}
    metadata=json.loads((a.run/'task.json').read_text())
    evaluation=json.loads((a.run/'evaluation.json').read_text())
    dt=float(d['dt']);rows=[]
    lo,hi=metadata['tool_down_angle_range_deg']
    for env in range(d['active'].shape[1]):
        active=d['active'][:,env].astype(bool);ids=np.flatnonzero(active)
        b=d['block'][active,env];tip=d['tip'][active,env];tool=d['tool'][active,env]
        goal=d['goal'][env]+d['origins'][env]
        lift=b[:,2]-d['origins'][env,2]-(.72+.0061)
        xy=np.linalg.norm(b[:,:2]-goal[:2],axis=-1)
        delta=tip-tool[:,:3]
        down=np.rad2deg(np.arctan2(-delta[:,2],np.linalg.norm(delta[:,:2],axis=-1)))
        normal=d['normal_angle'][active,env]
        face=d['face'][active,env].astype(bool)
        both=d['both'][active,env].astype(bool)
        retention=d['retained'][active,env]<.035
        orientation_error=d['orientation_error'][active,env]
        orient=d['orientation_ok'][active,env].astype(bool)
        other_orient=np.ones(len(ids),bool)
        if metadata.get('require_palm_down'):
            other_orient &= d['palm_normal'][active,env,2]<-.05
        if metadata.get('require_web_up'):
            other_orient &= d['web_axis'][active,env,2]>=metadata['min_web_up_component']
        if metadata.get('max_palm_up_component') is not None:
            other_orient &= d['palm_normal'][active,env,2]<=metadata['max_palm_up_component']
        criteria=metadata['success_definition']
        angle=(normal>=lo)&(normal<=hi)&(down>=lo)&(down<=hi)&orient
        rawlift=(lift>.045)&both
        heldlift=rawlift&face&retention&(orientation_error<.3)&other_orient
        final_conditions={
            'within_goal_xy':xy<criteria['final_xy_error_m'],
            'on_table':abs(lift)<criteria['table_height_error_m'],
            'low_linear_speed':np.linalg.norm(b[:,7:10],axis=-1)<criteria['linear_speed_m_s'],
            'low_angular_speed':np.linalg.norm(b[:,10:13],axis=-1)<criteria['angular_speed_rad_s'],
            'jaws_released':d['forces'][active,env].max(-1)<criteria['no_jaw_contact_N_max'],
            'tips_retracted':tip[:,2]-b[:,2]>criteria['tool_tip_above_block_m'],
            'tool_retained':retention,
            'wide_face_pinch':face,
            'relative_rotation_metric_below_0p3':orientation_error<.3,
            'palm_and_web_orientation_ok':other_orient,
        }
        placed=np.logical_and.reduce(list(final_conditions.values()))
        lifted_once=np.zeros(len(ids),bool);count=0;seen=False
        for n,v in enumerate(heldlift):
            count=count+1 if v else 0
            seen|=count>=round(criteria.get('continuous_bilateral_lift_s',.25)/dt)
            lifted_once[n]=seen
        physical_candidate=placed&lifted_once
        phases=d['phase'][active,env]
        phase_rows=[]
        for phase in np.unique(phases):
            m=phases==phase
            phase_rows.append(dict(phase=int(phase),name=metadata['phases'][int(phase)],duration_s=float(m.sum()*dt),
                first_s=float((ids[np.flatnonzero(m)[0]]+1)*dt),last_s=float((ids[np.flatnonzero(m)[-1]]+1)*dt),
                normal_down_angle_deg=[float(normal[m].min()),float(normal[m].max())],root_to_tip_down_angle_deg=[float(down[m].min()),float(down[m].max())],
                max_block_lift_m=float(lift[m].max()),min_goal_xy_error_m=float(xy[m].min()),bilateral_fraction=float(both[m].mean()),wide_face_fraction=float(face[m].mean())))
        rows.append(dict(environment=env,recorded_task_success=bool(evaluation['success_per_env'][env]),active_duration_s=len(ids)*dt,
            start_xy=d['start'][env,:2].tolist(),goal_xy=d['goal'][env,:2].tolist(),maximum_block_lift_m=float(lift.max()),
            maximum_block_xy_displacement_m=float(np.linalg.norm(b[:,:2]-b[0,:2],axis=-1).max()),
            longest_bilateral_lift_over_45mm_s=longest(rawlift,dt),longest_retained_face_bilateral_lift_without_angle_gate_s=longest(heldlift,dt),
            longest_lift_with_all_orientation_gates_s=longest(heldlift&angle,dt),
            final=dict(time_s=float((ids[-1]+1)*dt),phase=int(phases[-1]),block_xyz_m=(b[-1,:3]-d['origins'][env]).tolist(),goal_xy_error_m=float(xy[-1]),height_error_m=float(lift[-1]),linear_speed_m_s=float(np.linalg.norm(b[-1,7:10])),angular_speed_rad_s=float(np.linalg.norm(b[-1,10:13])),tip_clearance_above_block_center_m=float(tip[-1,2]-b[-1,2]),jaw_contact_forces_N=d['forces'][ids[-1],env].tolist(),normal_down_angle_deg=float(normal[-1]),root_to_tip_down_angle_deg=float(down[-1]),physical_conditions_without_angle_gate={k:bool(v[-1]) for k,v in final_conditions.items()},orientation_ok=bool(orient[-1])),
            angle=dict(required_deg=[lo,hi],normal_down_range_deg=[float(normal.min()),float(normal.max())],root_to_tip_down_range_deg=[float(down.min()),float(down.max())],normal_over_upper_s=float((normal>hi).sum()*dt),root_to_tip_over_upper_s=float((down>hi).sum()*dt),orientation_ok_fraction=float(orient.mean()),palm_and_web_ok_fraction=float(other_orient.mean()),max_relative_rotation_error_sin_half_angle_squared=float(orientation_error.max())),
            diagnostic_only=dict(longest_stable_placement_without_angle_gate_after_held_lift_s=longest(physical_candidate,dt),meets_stable_duration_without_angle_gate=longest(physical_candidate,dt)>=criteria['continuous_stability_s'],warning='Relaxed-gate diagnostic only; not the official task success criterion.'),
            wide_face_contact_fraction=float(face.mean()),phases=phase_rows))
    report=dict(run=str(a.run.resolve()),trajectory_sha256=hashlib.sha256((a.run/'trajectory.npz').read_bytes()).hexdigest(),
        method='Recomputed from saved active control samples: world body positions/velocities, bilateral jaw contacts, hand grip metrics and orientations. Recorded success flag is reported separately.',
        assumptions=dict(table_height_m=.72,resting_cube_center_above_table_m=.0061),rows=rows)
    output=a.out or a.run/'independent_motion_summary.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
