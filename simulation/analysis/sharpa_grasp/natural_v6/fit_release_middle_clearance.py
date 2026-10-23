"""Find a small middle-finger release offset from the actual supported C state.

Geometry only: no dynamic result is inferred. Source meshes and the existing
official mechanical exclusions are used, with no middle/tool exemption.
"""
from pathlib import Path
import argparse,copy,hashlib,itertools,json,sys,time
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'analysis/sharpa_grasp'))
from fit_natural_support import NaturalFit,module,transform

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=ROOT/'runs/sharpa_w01_natural_v6c_cold');p.add_argument('--out',type=Path,default=ROOT/'analysis/sharpa_grasp/natural_v6/release_middle_clearance');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 report=json.loads((a.run/'report.json').read_text());cfg=json.loads((ROOT/'configs/sharpa_w01_natural_v6c.json').read_text());f=NaturalFit();audit=f.audit
 with np.load(a.run/'timeseries.npz',allow_pickle=False)as d:q0=d['q'][-1,0].astype(float);b0=d['bodies'][-1,0].astype(float)
 assert audit.names==report['joint_names'];origin=np.asarray(report['env_origins'][0]);states0=dict(zip(['tool','left','right'],b0));delta=np.array([cfg['closure_delta'].get(n,0.)for n in audit.names]);names=['right_middle_'+n for n in ['MCP_AA','MCP_FE','PIP','DIP']];indices={n:audit.names.index(n)for n in names}
 def state(offset,closure=.75,fraction=1.,jaw_extra=0.):
  q=q0+(closure-.75)*delta
  for n,v in offset.items():q[indices[n]]+=fraction*v
  clipped=[]
  for i,n in enumerate(audit.names):
   el=audit.kin.root.find("joint[@name='"+n+"']/limit")
   if el is not None and 'lower'in el.attrib:
    v=float(np.clip(q[i],float(el.get('lower')),float(el.get('upper'))))
    if v!=q[i]:clipped.append({'joint':n,'unclipped_rad':float(q[i]),'clipped_rad':v});q[i]=v
  states=copy.deepcopy(states0)
  for key,sign in [('left',-1),('right',1)]:
   if jaw_extra:
    T=module.pose_transform(states[key]);T=T@transform([0,0,0],Rotation.from_euler('y',sign*jaw_extra).as_matrix());states[key][:3]=T[:3,3];states[key][3:7]=np.roll(Rotation.from_matrix(T[:3,:3]).as_quat(),1)
  bounds=audit.state(q,states,origin,True);mid=audit.examine(bounds,f.middle_pairs);ss=audit.examine(bounds,audit.self_pairs);ts=audit.examine(bounds,audit.nontarget_pairs)
  violations=[]
  for j in audit.kin.joints:
   if j['name']in audit.names and j['kind']not in ['fixed','continuous']:
    v=q[audit.names.index(j['name'])]
    # Source URDF limits are the authority, not fitted soft ranges.
    el=audit.kin.root.find("joint[@name='"+j['name']+"']/limit")
    if el is not None and not float(el.get('lower'))-1e-8<=v<=float(el.get('upper'))+1e-8:violations.append(j['name'])
  return {'closure':closure,'fraction':fraction,'jaw_extra_angle_rad':jaw_extra,'middle_gap_m':mid['min_distance_m'],'middle_pair':mid['nearest_pair'],'self_gap_m':ss['min_distance_m'],'nontarget_gap_m':ts['min_distance_m'],'self_intersections':ss['intersections'],'nontarget_intersections':ts['intersections'],'limit_violations':violations,'extrapolated_joint_limit_clipping':clipped}
 baseline=state({});print('BASELINE',json.dumps(baseline),flush=True);candidates=[];began=time.monotonic()
 for n,v in itertools.product(names,np.r_[np.arange(-.12,-.004,.005),np.arange(.005,.1201,.005)]):
  offset={n:float(v)};r=state(offset)
  if r['self_intersections']or r['nontarget_intersections']or r['limit_violations']or r['middle_gap_m']<.001:continue
  candidates.append({'offset_rad':offset,'norm_rad':abs(float(v)),'endpoint':r})
 candidates.sort(key=lambda c:c['norm_rad']);(a.out/'endpoint_candidates.json').write_text(json.dumps(candidates,indent=2)+'\n');print('ENDPOINTS',len(candidates),json.dumps(candidates[:6]),flush=True)
 accepted=[]
 for candidate in candidates[:18]:
  offset=candidate['offset_rad'];rows=[state(offset,float(c),float(frac))for c,frac in itertools.product([.55,.65,.75,.85,1.],np.linspace(0,1,11))]
  bad=[r for r in rows if r['self_intersections']or r['nontarget_intersections']or r['limit_violations']]
  min_endpoint=min(r['middle_gap_m']for r in rows if r['fraction']==1.)
  candidate.update(transition_samples=len(rows),transition_intersection_or_limit_states=len(bad),minimum_endpoint_middle_gap_m=min_endpoint,minimum_transition_self_gap_m=min(r['self_gap_m']for r in rows),transition_rows=rows)
  if not bad and min_endpoint>=.001:
   # Moving jaws are not kinematically commanded by closure. This is a separate
   # relative-angle sensitivity sweep about each actual recorded blade frame.
   extra=[state(offset,float(c),1.,float(j))for c,j in itertools.product([.55,.75,1.],[-.025,-.0125,0.,.0125,.025])]
   candidate['jaw_angle_sensitivity']=extra;accepted.append(candidate);print('ACCEPTED',json.dumps({k:v for k,v in candidate.items()if k not in ['transition_rows','jaw_angle_sensitivity']}),flush=True)
   if len(accepted)>=3:break
 result={'method':__doc__,'source_run':str(a.run),'timeseries_sha256':sha(a.run/'timeseries.npz'),'source_pose_sha256':sha(ROOT/'configs/sharpa_w01_natural_v6c.json'),'script_sha256':sha(__file__),'baseline':baseline,'baseline_middle_joint_rad':{n:float(q0[i])for n,i in indices.items()},'reference_closure':.75,'endpoint_candidates':len(candidates),'accepted':accepted,'wall_seconds':time.monotonic()-began,'limitations':['This is static source geometry at the actual cold final tool/body pose, not a physical release validation.','Closure shifts thumb/index joint values by the existing frozen closure_delta; jaw articulation remains an independent measured or sensitivity-swept transform.','Transition is checked at 11 uniform fractions for each of five closure values, not a continuous collision proof.','Jaw sensitivity rotates actual blade frames without changing their recorded positions; actual dynamic tool motion must be rechecked.']};out=a.out/'report.json';out.write_text(json.dumps(result,indent=2)+'\n');print('REPORT',out,flush=True)
if __name__=='__main__':main()
