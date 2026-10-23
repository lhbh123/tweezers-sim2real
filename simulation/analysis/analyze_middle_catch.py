"""Measure gravity slip and middle-finger reaction from a physical joint-only probe."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();r=json.load(open(a.run/'report.json'));d=np.load(a.run/'timeseries.npz');t=d['time'];orig=np.asarray(r['env_origins']);b=d['bodies'];com=np.zeros(b.shape[:2]+(3,))
for j,(mass,local) in enumerate([(.003,[0,0,-.005]),(.006,[0,0,(.165*.014*.0825+.015*.005*.1725)/(.165*.014+.015*.005)]),(.006,[0,0,(.165*.014*.0825+.015*.005*.1725)/(.165*.014+.015*.005)])]):
 q=b[:,:,j,3:7];offset=Rotation.from_quat(q.reshape(-1,4)[:,[1,2,3,0]]).apply(np.tile(local,(q.shape[0]*q.shape[1],1))).reshape(q.shape[:2]+(3,));com+=mass*(b[:,:,j,:3]+offset)
com/=.015
base=(t>.8)&(t<=1);loose=(t>=1.5)&(t<=3);late=(t>=2.5)&(t<=3);rows=[]
for i,c in enumerate(r['candidates']):
 drop=(com[base,i,2].mean()-com[:,i,2])*1000;vel=np.gradient(com[:,i,2],t);mid=d['role_support_z_N'][:,i,2];under=d['middle_under_contact_count'][:,i];normal=d['role_normal_force_w'][:,i,2,2] if 'role_normal_force_w' in d else None
 qualifying=loose&(mid>.03)&(under>0)
 row=dict(name=c['name'],baseline_middle_contact_fraction=float((d['role_contact_strength_N'][base,i,2]>.003).mean()),baseline_middle_up_N=float(mid[base].mean()),max_COM_drop_during_loose_mm=float(drop[loose].max()),late_COM_drop_mm=float(drop[late].mean()),late_COM_vertical_speed_mm_s=float(vel[late].mean()*1000),late_middle_net_up_N=float(mid[late].mean()),late_middle_normal_up_N=float(normal[late].mean()) if normal is not None else None,late_middle_under_contact_fraction=float((under[late]>0).mean()),support_duration_loose_s=float(qualifying.sum()*(t[1]-t[0])),final_broad_face_fraction=c['final_1s_both_assigned_broad_face_fraction'],final_retained=c['retained'],max_grip_error_m=float(d['grip_error_m'][loose,i].max()),min_tool_tip_clearance_m=float(d['tip_clearance_bound_m'][loose,i].min()))
 rows.append(row)
out=dict(experiment='Scripted thumb/index actuator loosening with passive middle backup; not an RL policy test',runtime_object_writes=r['runtime_object_state_writes'],tool_mass_kg=.015,COM_construction='Uniform density from authored 165x14x1 mm strip plus15x5x1 mm tip, total6g per jaw; joined handle3g at z=-5mm. No inertial override.',cases=rows)
(a.run/'catch_analysis.json').write_text(json.dumps(out,indent=2));print(json.dumps(rows,indent=2))
