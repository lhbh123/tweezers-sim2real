import sys,json
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from kinematics import Kinematics
from arm_command import ArmCommandKinematics
torch.set_num_threads(2);dev='cpu';kin=Kinematics();arm=ArmCommandKinematics(kin,dev)
meta=json.load(open(ROOT/'runs/pick_place15_nominal17/task.json'));ref=meta['arm_reset_IK_evidence'][0]['q'];q0=torch.tensor([[ref[n] for n in kin.arm]],dtype=torch.float32)
xy=torch.tensor([[x,y] for x in np.linspace(.48,.76,8) for y in np.linspace(-.22,.08,9)],dtype=torch.float32);n=len(xy)
q=q0.repeat(n,1);T,_=arm.fk_jacobian(q);home=torch.tensor([.48,0,.0828]);offset=home-T[0,:3,3];lower=T[:,:3,3]-torch.tensor([.35,.35,.09]);upper=T[:,:3,3]+torch.tensor([.35,.35,.1])
for height in [.0861,.0076,.0861,.0076,.0861]:
 target=torch.cat((xy,torch.full((n,1),height)),dim=-1)
 for step in range(800):
  now=arm.fk_jacobian(q)[0][:,:3,3]+offset;delta=(target-now)*.5;delta*=torch.minimum(torch.ones(n),.001/delta.norm(dim=-1).clamp_min(1e-9))[:,None]
  q=arm.increment(q,delta,T,lower,upper)
 now=arm.fk_jacobian(q)[0][:,:3,3]+offset;error=(target-now).norm(dim=-1)
 print('region',[.48,.76,-.22,.08],'height',height,'maxerror',float(error.max()),'failed',xy[error>.001].tolist(),flush=True)
