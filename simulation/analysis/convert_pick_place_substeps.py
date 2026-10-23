"""Losslessly expose env0 native substeps to the existing source-mesh auditor."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
r=json.loads((a.run/'evaluation.json').read_text());assert r['episodes']==1
z=np.load(a.run/'trajectory.npz');d={k:z[k] for k in z.files};z.close();n=len(d['q'])
with (a.run/'grasp_substeps.jsonl').open() as src,(a.run/'trajectory.json').open('w') as dst:
 dst.write('[')
 for k,line in enumerate(src):
  raw=json.loads(line);assert abs(raw['time']-(k+1)*float(d['dt']))<1e-8
  row=dict(time=raw['time'],joint_pos=d['q'][k,0].tolist(),cube=d['block'][k,0].tolist(),tool=d['tool'][k,0].tolist(),left=d['left'][k,0].tolist(),right=d['right'][k,0].tolist(),done=k==n-1,grasp_contact_substeps=raw['substeps'])
  for name,key in [('joint_pos','q'),('tool','tool'),('left','left'),('right','right')]:assert np.array_equal(np.array(raw['substeps'][-1][name]),d[key][k,0]),f'{k} {name}: mismatched recording'
  if k:dst.write(',')
  json.dump(row,dst)
 assert k+1==n
 dst.write(']')
print('Converted without changing any recorded state:',n,'control steps',n*r['task']['decimation'],'physics samples')
