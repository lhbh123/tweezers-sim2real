"""Task and independent-auditor regressions, not a physical success test."""
import json
import numpy as np
import torch
from check_rl_stateflow import BufferScene, BufferTask, TaskConfig
from audit_recorded_face_pinch import point_values


def check_task(device):
    scene=BufferScene(device)
    scene.state['palm_quat_w']=scene.state['wrist_quat_w']
    support={k:torch.zeros(shape,device=device) for k,shape in {
        'role_contact_strength_N':(3,5),'role_support_z_N':(3,5),
        'web_normal_force_w':(3,3),'web_friction_force_w':(3,3),
        'web_contact_count':(3,),'middle_under_contact_count':(3,),
        'broad_face_inward_force_N':(3,2,2),'broad_face_contact_count':(3,2,2)}.items()}
    support['broad_face_inward_force_N'][:,1,0]=.04
    support['broad_face_inward_force_N'][:,0,1]=.05
    support['broad_face_contact_count'][:,1,0]=1
    support['broad_face_contact_count'][:,0,1]=1
    scene.read_grasp_support=lambda state:support
    cfg=TaskConfig(reset_xy_range=0,require_face_pinch=True)
    task=BufferTask(scene,cfg)
    assert task.measurements()['face_pinch_ok'].all(), 'Middle/web absence must be allowed'
    action=torch.zeros((3,4),device=device)
    _,reward0,_,_=task.step(action,auto_reset=False)
    task.reset()
    support['role_contact_strength_N'][:,2]=1
    support['role_support_z_N'][:,2]=1
    support['web_contact_count'][:]=1
    support['web_normal_force_w'][:,2]=1
    _,reward1,_,_=task.step(action,auto_reset=False)
    torch.testing.assert_close(reward0,reward1)
    support['broad_face_inward_force_N'][0,1,0]=0
    support['broad_face_inward_force_N'][0,0,0]=2 # wrong blade does not count
    support['broad_face_contact_count'][1,0,1]=0
    assert task.measurements()['face_pinch_ok'].tolist()==[False,False,True]
    # Mere edge pressure is absent from qualifying-force tensors, so cannot pass.
    support['broad_face_inward_force_N'][2]=0
    assert not task.measurements()['face_pinch_ok'].any()


def check_geometry():
    geo=dict(length_m=.16,strip_width_m=.012,strip_thickness_m=.001,tip_length_m=.015)
    crit=dict(width_margin_m=.002,axial_margin_m=.003,face_tolerance_m=.0004,min_inward_cos=.8660254)
    for sign in (-1,1):
        p=np.array([sign*.0005,0,.07]);f=np.array([-sign*.1,0,0])
        np.testing.assert_allclose(point_values(p,f,sign,geo,crit),[.1,1,.007])
        for badp,badf in [(p+[0,.0059,0],f),(-p,f),(p,-f),
                          (p,np.array([0,.1,0])),(p+[0,0,.085],f)]:
            assert not point_values(badp,badf,sign,geo,crit).any()


if __name__=='__main__':
    torch.set_num_threads(2)
    devices=['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])
    for device in devices:check_task(device)
    check_geometry()
    print(json.dumps(dict(devices=devices,optional_middle_web_absence='pass',
        no_middle_web_reward='pass',opposing_blade_assignment='pass',
        no_edge_or_wrong_direction_acceptance='pass',positive_closing_torque='pass',
        physical_success_tested=False)))
