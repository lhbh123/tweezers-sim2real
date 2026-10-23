"""Independent NumPy FK / finite differences plus exact-zero command regression."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from kinematics import Kinematics
from arm_command import ArmCommandKinematics


def check(device):
    k=Kinematics();m=ArmCommandKinematics(k,device,torch.float64)
    q=torch.tensor([[.14259,-1.25138,2.37364,-.69390,2.80401,1.67209],
                    [.1,-.5,1.5,-1.4,1.2,.2]],dtype=torch.float64,device=device)
    T,J=m.fk_jacobian(q)
    independent=np.array([k.fk(dict(zip(k.arm,row)))['wrist_3_link'] for row in q.cpu().numpy()])
    fk_error=float(np.max(np.abs(T.cpu().numpy()-independent)))
    derivatives=[]
    for j in range(6):
        perturbed=q.cpu().numpy().copy();perturbed[:,j]+=1e-6
        second=np.array([k.fk(dict(zip(k.arm,row)))['wrist_3_link'] for row in perturbed])
        linear=(second[:,:3,3]-independent[:,:3,3])/1e-6
        angular=np.array([Rotation.from_matrix(a@b.T).as_rotvec()/1e-6
                          for a,b in zip(second[:,:3,:3],independent[:,:3,:3])])
        derivatives.append(np.concatenate((linear,angular),axis=-1))
    jac_error=float(np.max(np.abs(np.stack(derivatives,-1)-J.cpu().numpy())))
    p=T[:,:3,3]
    stationary=m.increment(q,torch.zeros((2,3),device=device),T,p-.2,p+.2)
    assert torch.equal(stationary,q), 'Zero command erased or changed actuator target'
    delta=torch.tensor([[.001,0,0],[0,0,.001]],dtype=q.dtype,device=device)
    next_q=m.increment(q,delta,T,p-.2,p+.2)
    after,_=m.fk_jacobian(next_q)
    position_error=float((after[:,:3,3]-p-delta).abs().max())
    rotation_error=float((after[:,:3,:3]-T[:,:3,:3]).abs().max())
    bounded=m.increment(q,delta*1000,T,p-.2,p+.2)
    assert float((bounded-q).abs().max())<=.025+1e-12
    assert bool(((bounded>=m.lo)&(bounded<=m.hi)).all())
    assert fk_error<1e-12 and jac_error<1e-6 and position_error<1e-5 and rotation_error<1e-5
    return dict(device=device,FK_max_error=fk_error,finite_difference_J_error=jac_error,
                exact_zero_hold=True,millimeter_command_position_error_m=position_error,
                rotation_matrix_error=rotation_error,joint_speed_and_limit_bounds=True)


if __name__=='__main__':
    devices=['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])
    report={'tests':[check(d) for d in devices],
            'scope':'Kinematic arithmetic and actuator target semantics; not physical grasp validation.'}
    (ROOT/'reports/arm_command_regression.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
