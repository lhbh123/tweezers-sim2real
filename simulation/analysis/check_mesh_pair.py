"""Continuous triangle-pair SAT narrow phase for one source-mesh pair."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from kinematics import ASSET,Kinematics,stl_triangles

def triangle_pair_overlap(a,b):
    ea=np.stack([a[:,1]-a[:,0],a[:,2]-a[:,1],a[:,0]-a[:,2]],axis=1)
    eb=np.stack([b[:,1]-b[:,0],b[:,2]-b[:,1],b[:,0]-b[:,2]],axis=1)
    na=np.cross(ea[:,0],ea[:,1]);nb=np.cross(eb[:,0],eb[:,1])
    axes=[na,nb]
    axes.extend(np.cross(ea[:,i],eb[:,j]) for i in range(3) for j in range(3))
    # Additional in-plane axes handle coplanar triangles.
    axes.extend(np.cross(na,ea[:,i]) for i in range(3))
    axes.extend(np.cross(nb,eb[:,i]) for i in range(3))
    intersects=np.ones(len(a),dtype=bool)
    for axis in axes:
        pa=np.einsum('nij,nj->ni',a,axis);pb=np.einsum('nij,nj->ni',b,axis)
        intersects&=(pa.max(axis=1)>=pb.min(axis=1)-1e-13)&(pb.max(axis=1)>=pa.min(axis=1)-1e-13)
    return intersects

def mesh_intersections(a,b):
    amin,amax=a.min(axis=1),a.max(axis=1);bmin,bmax=b.min(axis=1),b.max(axis=1)
    broad_a=np.all((amax>=bmin.min(axis=0))&(amin<=bmax.max(axis=0)),axis=1)
    hits=[];tested=0
    for i in np.flatnonzero(broad_a):
        candidates=np.flatnonzero(np.all((bmax>=amin[i])&(bmin<=amax[i]),axis=1))
        if not len(candidates):continue
        tested+=len(candidates)
        valid=triangle_pair_overlap(np.repeat(a[i][None,:,:],len(candidates),axis=0),b[candidates])
        hits.extend((int(i),int(j)) for j in candidates[valid])
    return {'triangle_pair_tests':tested,'intersection_pair_count':len(hits),'intersection_pairs':hits}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/human_grasp_precision_v2.json')
    p.add_argument('--link-a',default='right_hand_link');p.add_argument('--link-b',default='right_hand_index_rota_link1')
    p.add_argument('--output',type=Path,default=ROOT/'analysis/human_grasp_v2_palm_index_narrowphase.json')
    p.add_argument('--zero',action='store_true');args=p.parse_args()
    k=Kinematics();q={} if args.zero else json.loads(args.config.read_text())['q_reference'];Ts=k.fk(q,root='right_hand_link')
    meshes=[]
    for name in [args.link_a,args.link_b]:
        tri=stl_triangles(ASSET/'sources/spider/assets'/f'{name}.STL');T=Ts[name]
        meshes.append(tri@T[:3,:3].T+T[:3,3])
    result=mesh_intersections(*meshes)
    result.update({'config':str(args.config),'q_zero':args.zero,'links':[args.link_a,args.link_b],
                   'method':'Triangle pair SAT with AABB broad phase and coplanar in-plane axes',
                   'limitation':'Surface intersections only; fully contained disjoint shells require an additional volume containment test.'})
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='intersection_pairs'},indent=2))
