"""CPU checks: contact witness boundary cases and independent saved-point replay.

Run with Isaac's Python for its CPU torch dependency; no app or CUDA is started.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='src/grasp_contacts.py')
    parser.add_argument('--runs',nargs='+',required=True)
    parser.add_argument('--out',required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('review_grasp_contacts',args.source)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reader = module.TripodContactReader.__new__(module.TripodContactReader)
    reader.face_geometry = {'thickness_m':.001,'width_m':.014,'length_m':.14,'tip_length_m':.015}
    reader.face_config = {'width_margin_m':.002,'axial_margin_m':.003,'face_tolerance_m':.0004,
                          'min_inward_cos':.8660254,'allow_tip_contact':False}
    reader.face_contact_point_mode='validated_tool_witness';reader.tool_witness_plane_tolerance_m=20e-6
    T = lambda x:torch.tensor(x,dtype=torch.float64)
    normal=T([-1.,0,0]);force=T([.2,0,0]);sep=T(-.0007)
    checks=[]
    def check(name,value):
        assert bool(value),name;checks.append(name)
    point=T([.0002,0,.04])
    m=reader.face_contact_metrics(point,force,-1,True,normal,sep)
    check('deep valid contact reconstructs to outer plane',m['tool_surface_witness_used'] and m['broad_face_qualified'])
    check('raw coordinates preserved',torch.equal(m['point_jaw_local_m'],point))
    check('normal torque invariant',abs(float(m['witness_closing_torque_difference_Nm']))<1e-14)
    reader.face_contact_point_mode='raw'
    m=reader.face_contact_metrics(point,force,-1,True,normal,sep)
    check('raw compatibility retains prior rejection',not m['tool_surface_witness_used'] and not m['broad_face_qualified'])
    reader.face_contact_point_mode='validated_tool_witness'
    for name,p,f,n,s,normal_contact in [
        ('wrong separation cannot rescue',point,force,normal,T(-.0005),True),
        ('positive scalar cannot rescue',point,force,-normal,sep,True),
        ('nonunit normal cannot rescue',point,force,2*normal,sep,True),
        ('nonfinite separation cannot rescue',point,force,normal,T(float('nan')),True),
        ('friction never qualifies as normal',point,force,normal,sep,False),
        ('too near width edge rejected',T([.0002,.006,.04]),force,normal,sep,True),
        ('too near axial end rejected',T([.0002,0,.001]),force,normal,sep,True),
        ('tip contact remains excluded',T([.0002,0,.13]),force,normal,sep,True),
        ('outward force rejected',point,-force,normal,sep,True),
    ]:
        m=reader.face_contact_metrics(p,f,-1,normal_contact,n,s)
        check(name,not m['broad_face_qualified'])
    m=reader.face_contact_metrics(T([-.0002,0,.04]),-force,1,True,-normal,sep)
    check('right jaw symmetry',m['tool_surface_witness_used'] and m['broad_face_qualified'])
    m=reader.face_contact_metrics(torch.empty((0,3)),torch.empty((0,3)),-1,True,
                                  torch.empty((0,3)),torch.empty((0,1)))
    check('empty contact batches supported',m['broad_face_qualified'].numel()==0)
    replay=[]
    for run_text in args.runs:
        run=Path(run_text);report=json.loads((run/'report.json').read_text())
        data=np.load(run/'timeseries.npz');times=data['time'].ravel()
        reader.face_geometry=report['face_definition']['geometry'];reader.face_config=report['face_definition']['criteria']
        recorded_mode=report.get('face_definition',{}).get('contact_point_mode','raw')
        reader.face_contact_point_mode='validated_tool_witness'
        npoints=0;classification_mismatch=0;valid_count=0;newly_qualified=0;torque_diff=0.;raw_mode_mismatch=0
        for line in (run/'raw_contact_points.jsonl').open():
            row=json.loads(line);idx=int(np.argmin(abs(times-row['time_s'])));env=row['env']
            for point in row['points']['normal_contacts']:
                if point['tool'] not in ('left','right') or point['role'] not in ('thumb','index'):continue
                bi=['left','right'].index(point['tool'])+1;sign=-1 if bi==1 else 1
                body=data['bodies'][idx,env,bi]
                n=Rotation.from_quat(np.roll(body[3:7],-1)).inv().apply(point['normal_world'])
                p=np.asarray(point['point_jaw_local_m']);f=np.asarray(point['force_jaw_local_N']);s=point['separation_m']
                # Independent NumPy classification, including all old face gates.
                candidate=p-s*n;geo=reader.face_geometry;c=reader.face_config
                valid=(np.isfinite(candidate).all() and np.dot(f,n)<-1e-8 and abs(np.linalg.norm(n)-1)<=1e-3
                       and abs(sign*candidate[0]-.5*geo['thickness_m'])<=20e-6)
                chosen=candidate if valid else p
                tip=chosen[2]>geo['length_m']-geo['tip_length_m']
                width=min(.005,geo['width_m']) if tip else geo['width_m']
                z0=geo['length_m']-geo['tip_length_m'] if tip else 0.
                z1=geo['length_m'] if tip else geo['length_m']-geo['tip_length_m']
                inward=-sign*f[0]
                qualified=bool(width/2-abs(chosen[1])>=c['width_margin_m'] and
                    min(chosen[2]-z0,z1-chosen[2])>=c['axial_margin_m'] and
                    abs(sign*chosen[0]-geo['thickness_m']/2)<=c['face_tolerance_m'] and
                    inward/max(np.linalg.norm(f),1e-12)>=c['min_inward_cos'] and inward>1e-8 and
                    (c['allow_tip_contact'] or not tip))
                m=reader.face_contact_metrics(T(p),T(f),sign,True,T(n),T(s))
                classification_mismatch+=int(bool(m['broad_face_qualified'])!=qualified or bool(m['tool_surface_witness_valid'])!=valid)
                valid_count+=int(valid);newly_qualified+=int(qualified and not point['broad_face_qualified'])
                torque_diff=max(torque_diff,abs(float(m['witness_closing_torque_difference_Nm'])))
                reader.face_contact_point_mode='raw'
                raw=reader.face_contact_metrics(T(p),T(f),sign,True,T(n),T(s))
                compare=raw if recorded_mode=='raw' else m
                raw_mode_mismatch+=int(bool(compare['broad_face_qualified'])!=point['broad_face_qualified'])
                reader.face_contact_point_mode='validated_tool_witness';npoints+=1
        check(run.name+' independent classification agreement',classification_mismatch==0)
        check(run.name+' recorded '+recorded_mode+' classification agreement',raw_mode_mismatch==0)
        check(run.name+' normal moment invariant within 1e-8 Nm',torque_diff<1e-8)
        replay.append({'run':str(run.resolve()),'contact_count':npoints,'valid_witness_count':valid_count,
                       'newly_qualified_count':newly_qualified,'classification_mismatch':classification_mismatch,
                       'recorded_mode':recorded_mode,'recorded_mode_classification_mismatch':raw_mode_mismatch,
                       'max_normal_torque_difference_Nm':torque_diff,
                       'raw_sha256':hashlib.sha256((run/'raw_contact_points.jsonl').read_bytes()).hexdigest()})
    result={'all_passed':True,'source_sha256':hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
            'device':'CPU only','checks':checks,'replayed_runs':replay}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
