"""CPU regression: explicit world orientation must affect the snapshot reset.

Checks actual robot FK and all three dynamic bodies. This is kinematic testing,
not evidence that a rotated grasp remains stable under gravity.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from reset_reference import StableResetReference

source=ROOT/'runs/natural_hold_distal_open060/stable_snapshot.json'
reference=StableResetReference(source)
p=np.deg2rad(40.)
axis=np.array([np.cos(p),0.,-np.sin(p)])
across=np.array([0.,-1.,0.])
desired=np.column_stack([across,np.cross(axis,across),axis])
points=np.array([[.48+dx,dy,.7478] for dx,dy in [(0,0),(-.002,0),(.002,0),(0,-.002),(0,.002)]])
built=reference.build(points,env_origins=np.zeros((5,3)),position_frame='local',desired_tip_orientation=desired)
for i in range(5):
    state=built['object_states']['handle'][i].numpy()
    matrix=Rotation.from_quat(np.r_[state[4:7],state[3]]).as_matrix()
    np.testing.assert_allclose(matrix,desired,atol=2e-7)
    q=dict(zip(reference.joint_names,built['q_actual'][i].numpy()))
    fk=reference.kin.fk(q)
    palm_normal=fk['right_hand_link'][:3,:3]@np.array([0.,1.,0.])
    assert palm_normal[2]<-.6, palm_normal
    tips=[]
    for name in ['left','right']:
        jaw=built['object_states'][name][i].numpy()
        R=Rotation.from_quat(np.r_[jaw[4:7],jaw[3]]).as_matrix()
        tips.append(jaw[:3]+R@reference.tip_local)
    np.testing.assert_allclose(np.mean(tips,axis=0),points[i],atol=2e-7)
    wrist=fk['wrist_3_link'].copy()
    wrist[:3,3]+=reference.base_position
    observed=wrist@reference.wrist_to_tool
    np.testing.assert_allclose(observed[:3,:3],matrix,atol=3e-7)
    np.testing.assert_allclose(observed[:3,3],state[:3],atol=2e-7)
assert np.max(built['ik_error'].numpy())<1e-4
result={'explicit_snapshot_orientation':'pass','robot_FK_matches_rotated_tool':'pass',
        'both_jaw_tip_centres_match_reset':'pass','palm_z':float(palm_normal[2]),
        'tool_down_angle_deg':40.,'dynamic_stability_tested':False}
(ROOT/'reports/orientation_reset_regression.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
