"""Audit raw physics-step contact records without running or modifying simulation."""
import argparse
import json
from pathlib import Path
import numpy as np


def audit(run):
    report=json.loads((run/'report.json').read_text())
    d=np.load(run/'support_timeseries.npz')
    dt=float(report['sampling_dt_s'])
    if abs(dt-float(report['arguments']['dt']))>1e-9:
        raise ValueError('Requires records from every physics step; downsampled input aliases contact forces')
    weight=sum(report['tool_body_masses_kg'])*9.81
    strength=d['group_strength'][...,:3]
    role_fz=(d['group_normal']+d['group_friction'])[...,2]
    web_fz=d['web_normal'][...,2]+d['web_friction'][...,2]
    web_hit=(d['web_hits']>0).astype(float)
    mid_under=d['middle_under'].astype(float)
    times=d['time'];n=strength.shape[1]
    output={'run':str(run.resolve()),'physics_dt_s':dt,'tool_weight_N':weight,
            'threshold_N':.003,'object_state_writes':report['task_time_object_state_writes'],
            'definition':'Nonoverlapping windows average every physics substep. Three finger strengths >3mN, middle and Handle-web net upward mean >3mN, at least one web ROI point and one middle-below-axis point. Window mean support does not imply uninterrupted individual-substep contact.',
            'windows':{},'per_phase_alias_check':[]}
    for size in [1,4,8,16]:
        k=len(times)//size
        def window(x):return x[:k*size].reshape(k,size,*x.shape[1:]).mean(1)
        t=window(times);st=window(strength);rf=window(role_fz)
        wf=window(web_fz);wh=window(web_hit);mu=window(mid_under)
        gate=(st.min(-1)>.003)&(rf[:,:,2]>.003)&(wf>.003)&(wh>0)&(mu>0)
        periods={}
        for name,mask in [('after_1s',t>=1.),('last_1s',t>=times[-1]-1.)]:
            env=[]
            for i in range(n):
                env.append({'env':i,'support_window_fraction':float(gate[mask,i].mean()),
                    'mean_finger_strength_N':st[mask,i].mean(0).tolist(),
                    'middle_upward_mean_N':float(rf[mask,i,2].mean()),
                    'middle_upward_min_window_N':float(rf[mask,i,2].min()),
                    'web_upward_mean_N':float(wf[mask,i].mean()),
                    'web_upward_min_window_N':float(wf[mask,i].min()),
                    'web_contact_mean_duty_fraction':float(wh[mask,i].mean()),
                    'middle_under_mean_duty_fraction':float(mu[mask,i].mean()),
                    'total_upward_over_weight_mean':float(rf[mask,i].sum(-1).mean()/weight),
                    'total_upward_over_weight_std':float(rf[mask,i].sum(-1).std()/weight)})
            periods[name]=env
        output['windows'][str(size)]={'duration_s':size*dt,'periods':periods}
    tail=times>=times[-1]-1.
    for phase in range(8):
        selected=tail&(np.arange(len(times))%8==phase)
        output['per_phase_alias_check'].append({'phase':phase,
            'total_upward_over_weight':(role_fz[selected].sum(-1).mean(0)/weight).tolist(),
            'middle_upward_N':role_fz[selected,:,2].mean(0).tolist(),
            'web_upward_N':web_fz[selected].mean(0).tolist()})
    path=run/'support_window_audit.json'
    path.write_text(json.dumps(output,indent=2))
    print(path)
    for size,entry in output['windows'].items():
        print(size,'last1',[round(e['support_window_fraction'],4) for e in entry['periods']['last_1s']],
            'after1',[round(e['support_window_fraction'],4) for e in entry['periods']['after_1s']])
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    audit(args.run)
