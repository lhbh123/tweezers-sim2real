"""Offline URDF FK/IK and exact source-vertex table-clearance workspace scan."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from reset_reference import StableResetReference
from arm_clearance import ArmTableClearance
r=StableResetReference(ROOT/'runs/middle_backup13_level_search/m9_c0_snapshot.json',tip_local=[0,0,.1725])
c=ArmTableClearance(r.kin);seed=r.q_actual[r.arm_indices]
rows=[]
for x in np.linspace(.32,.68,10):
 for y in np.linspace(-.24,.24,13):
  for z in [.728,.758,.808,.848]:
   T=r.wrist_transform.copy();T[:3,3]+=[x-r.tip_position[0],y-r.tip_position[1],z-r.tip_position[2]];T[2,3]-=.72
   sol=r.kin.inverse_arm(T,seed,frame='wrist_3_link');q=[sol['q'][n] for n in r.kin.arm]
   clear=c.clearance(q);ok=sol['error']<1e-4 and min(clear.values())>=.015
   rows.append(dict(tip=[float(x),float(y),z],ok=ok,error=sol['error'],clearance=min(clear.values()),q=q))
print('valid',sum(r['ok'] for r in rows),'/',len(rows))
for x in np.linspace(.32,.68,10):
 valid=[round(y,2) for y in np.linspace(-.24,.24,13) if all(r['ok'] for r in rows if abs(r['tip'][0]-x)<1e-6 and abs(r['tip'][1]-y)<1e-6)]
 print(round(x,2),valid,flush=True)
(ROOT/'reports/pick_place_workspace_scan.json').write_text(json.dumps(dict(method='IK plus source vertex arm/table clearance; not dynamic validation',rows=rows),indent=2))
