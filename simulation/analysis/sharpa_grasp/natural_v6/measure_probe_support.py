"""Measure final recorded middle backup geometry in the actual probe gravity frame."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'analysis/sharpa_grasp'))
from fit_natural_support import NaturalFit,transform,module

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();r=json.loads((a.run/'report.json').read_text())
 with np.load(a.run/'timeseries.npz',allow_pickle=False) as d:q=d['q'][-1];bodies=d['bodies'][-1];t=float(d['time'][-1])
 f=NaturalFit()
 if r['joint_names']!=f.audit.names:raise ValueError('Joint order mismatch')
 rows=[]
 for env in range(len(q)):
  states=dict(zip(['tool','left','right'],bodies[env]));orig=np.asarray(r['env_origins'][env]);f.audit.state(q[env],states,orig,True)
  qs=dict(zip(f.audit.names,q[env]));fk=f.kin.fk(qs);W=transform(orig+[0,0,.72])@fk[f.kin.palm_link]
  f.Wpalm=W[:3,:3];f.T=np.linalg.inv(W)@module.pose_transform(states['tool'])
  s=f.support(qs);s.update(environment=env,candidate=r['candidates'][env]['name'],time_s=t)
  rows.append(s)
 out=a.run/'final_middle_support_geometry.json';out.write_text(json.dumps({'method':'Original source collision meshes, actual final q and tool poses; world Z is actual gravity vertical, not preferred static orientation','timeseries_sha256':hashlib.sha256((a.run/'timeseries.npz').read_bytes()).hexdigest(),'samples':rows},indent=2)+'\n')
 print(json.dumps({'output':str(out),'summary':[{'candidate':s['candidate'],'gap_mm':s['distance_m']*1000,'witness_below_index_mm':s['middle_witness_below_index_contact_m']*1000,'DP_center_below_index_mm':s['middle_DP_center_below_index_DP_center_m']*1000,'vertical_gap_mm':s['vertical_gap_m']*1000} for s in rows]},indent=2))

if __name__=='__main__':main()
