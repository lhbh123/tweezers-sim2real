"""Data-integrity checks for the tripod compositor, no simulation."""
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from compose_tripod_comparison import contact_values,web_source_scope,temporal_scope,selected_environment_evidence

normal=[[0,0,.1],[0,0,.2],[0,0,-.3],[0,0,0],[0,0,0]]
friction=[[0,0,0],[0,0,0],[0,0,.8],[0,0,0],[0,0,0]]
r={'grasp_support':{'role_normal_force_w':normal,'role_friction_force_w':friction,
   'role_contact_strength_N':[.1,.2,.3,0,0],'role_support_z_N':[.1,.2,.5,0,0],
   'web_normal_force_w':[0,0,-.02],'web_friction_force_w':[0,0,.005],
   'web_contact_count':1,'middle_under_contact_count':1}}
a=contact_values(r)
assert abs(a['middle_up_N']-.5)<1e-10 and abs(a['web_up_N']+.015)<1e-10
assert contact_values({})['middle_up_N'] is None
bad=copy.deepcopy(r);bad['grasp_support']['role_support_z_N'][2]=-.5
try:contact_values(bad)
except ValueError:pass
else:raise AssertionError('Support sign mismatch was accepted')
bad=copy.deepcopy(r);bad['grasp_support']['role_contact_strength_N'][0]=float('nan')
try:contact_values(bad)
except ValueError:pass
else:raise AssertionError('Nonfinite contact value was accepted')
old=web_source_scope(ROOT/'runs/tripod_seed57_pose_preview');assert old['scope']=='all_tool_bodies'
with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)
    for name in ['grasp_contacts.py','rl_scene.py']:(p/name).write_text((ROOT/'src'/name).read_text())
    current=web_source_scope(p);assert current['scope']=='handle'
assert temporal_scope({'physics_dt':1/480,'decimation':8,'grasp_support_temporal_aggregation':'mean_over_every_physics_substep_in_control_interval'})['label']=='16.7 ms 控制周期均值'
assert not temporal_scope({})['is_control_interval_mean']
result={'checks_passed':7,'checks':['Signed middle support combines negative normal + positive friction','Signed web support preserves downward negative value','Missing support remains unknown, not zero','Inconsistent recorded support rejected','NaN contact rejected','Old all-tool vs new Handle-only ROI scope distinguished','Control-interval averaging labelled from recorded metadata, never assumed'], 'current_source_scope':current['scope'],'old_preview_scope':old['scope']}
assert selected_environment_evidence({'episodes':1,'successes':1})['selected_success_episode']
assert not selected_environment_evidence({'episodes':1,'successes':0})['selected_success_episode']
assert not selected_environment_evidence({'episodes':64,'successes':13})['selected_success_episode']
assert selected_environment_evidence({'episodes':4,'successes':2,'success_per_env':[True,False,True,False]})['selected_success_episode']
assert not selected_environment_evidence({'episodes':4,'successes':2,'success_per_env':[False,True,True,False]})['selected_success_episode']
for bad in [{'episodes':4,'successes':1,'success_per_env':[True]},
            {'episodes':2,'successes':1,'success_per_env':[1,0]},
            {'episodes':2,'successes':1,'success_per_env':[True,True]}]:
    try:selected_environment_evidence(bad)
    except ValueError:pass
    else:raise AssertionError('Ambiguous/inconsistent environment success accepted')
result['checks']+=['Legacy one-episode success remains supported','Population successes alone never certify recorded env0','Multi-env success requires explicit boolean env0=True','Mismatched length, nonboolean entries and aggregate inconsistency rejected']
result['checks_passed']=len(result['checks'])
(ROOT/'reports/tripod_compositor_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
