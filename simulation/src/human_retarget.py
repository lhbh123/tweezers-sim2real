"""Retarget measured human hand landmarks to a natural XHand precision grasp.

Human evidence and geometric contact fitting are saved separately.  A missing
video estimate is an error: this module never labels a hand-authored prior as a
pose inferred from video.  The supplied robot's morphology has fewer joints
than a human hand, so contact-constrained refinement is explicitly recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from kinematics import ASSET, Kinematics, stl_triangles, transform

ROOT = Path(__file__).resolve().parents[1]
FINGER_POINTS = {
    'thumb': [1, 2, 3, 4], 'index': [5, 6, 7, 8],
    'mid': [9, 10, 11, 12], 'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20],
}


def unit(v):
    return np.asarray(v, float) / max(np.linalg.norm(v), 1e-12)


def angle(a, b):
    return float(np.arccos(np.clip(unit(a) @ unit(b), -1., 1.)))


def natural_soft_bounds(k):
    """Task-specific comfort envelope, strictly inside this asset's limits."""
    bounds = {}
    for n in k.hand:
        if n == 'right_hand_thumb_bend_joint': low, high = .20, 1.40
        elif n == 'right_hand_thumb_rota_joint1': low, high = -.65, 1.42
        elif n == 'right_hand_thumb_rota_joint2': low, high = .05, 1.55
        elif n == 'right_hand_index_bend_joint': low, high = -.12, .12
        elif '_index_' in n: low, high = .10, 1.45
        elif '_mid_' in n: low, high = .10, 1.35
        else: low, high = .10, 1.25
        bounds[n] = [max(low, k.by_name[n]['lower']+.04),
                     min(high, k.by_name[n]['upper']-.04)]
    return bounds


def canonical_landmarks(points):
    """Normalize the anatomical palm, including an explicit left/right mirror.

    +X points from little-finger MCP to index MCP; -Z points from wrist toward
    middle MCP. +Y is chosen toward observed finger flexion. This may reflect
    left-hand evidence to the right XHand; the reflection is not a camera pose.
    """
    p = np.asarray(points, float)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        raise ValueError('Expected 21 finite 3-D landmarks')
    z = -unit(p[9]-p[0])
    x = unit((p[5]-p[17])-z*((p[5]-p[17]) @ z))
    y = unit(np.cross(z, x))
    bends = sum(p[j+1]-p[j] for j in [5, 9, 13, 17])
    reflected = bool(bends @ y < 0)
    if reflected: y = -y
    basis = np.column_stack([x, y, z])
    scale = float(np.linalg.norm(p[9]-p[0]))
    if scale < 1e-6: raise ValueError('Degenerate palm landmarks')
    return (p-p[0]) @ basis / scale, {
        'world_to_palm_basis_columns': basis.tolist(),
        'wrist_middle_mcp_length_input_units': scale,
        'reflection_to_positive_flexion': reflected,
    }


def landmark_joint_prior(frame, k, bounds):
    points = frame.get('landmarks_world_m')
    source = 'landmarks_world_m'
    if points is None:
        points = frame.get('landmarks_normalized')
        source = 'landmarks_normalized'
    p, canonical = canonical_landmarks(points)
    q = {}; measurements = {}
    use_2d = frame.get('world_3d_usable_for_strong_supervision') is False
    projected = frame.get('joint_bend_proxy_2d_rad', {})
    for finger, ids in FINGER_POINTS.items():
        bones = [p[ids[i+1]]-p[ids[i]] for i in range(3)]
        pip = angle(bones[0], bones[1]); dip = angle(bones[1], bones[2])
        mcp = max(0., float(np.arctan2(bones[0][1], -bones[0][2])))
        if use_2d:
            proxy=projected.get('middle' if finger=='mid' else finger)
            if proxy is None:raise ValueError('Rejected depth estimate needs an explicit pixel-angle proxy')
            mcp,pip,dip=map(float,proxy)
        measurements[finger] = {'mcp_flexion_rad':mcp,
                                'pip_flexion_rad':pip, 'dip_flexion_rad':dip}
        if finger == 'thumb':
            # The robot thumb's first joint is an opposition joint rather than
            # an anatomical MCP. Preserve the observed thumb ray direction and
            # let the geometric stage adapt its nonhuman kinematic chain.
            thumb_names=k.hand[:3]
            if use_2d:
                # The thumb looks nearly straight in this camera projection;
                # its out-of-plane opposition cannot be inferred reliably.
                # Keep a disclosed weak geometric prior rather than importing
                # the rejected MediaPipe depth flip as ground truth.
                q[thumb_names[0]]=.75*.35+.25*mcp
                q[thumb_names[1]]=.75*1.00+.25*pip
                q[thumb_names[2]]=.75*.50+.25*dip
                measurements[finger]['occluded_depth_regularization']='0.25 pixel-angle proxy + 0.75 geometric robot prior (.35,1,.5); not a measured 3-D thumb posture'
                continue
            target_dirs=np.array([unit(v) for v in bones])
            thumb_links=['right_hand_thumb_bend_link','right_hand_thumb_rota_link1',
                         'right_hand_thumb_rota_link2','right_hand_thumb_rota_tip']
            def thumb_residual(a):
                Ts=k.fk(dict(zip(thumb_names,a)),root='right_hand_link')
                positions=[Ts[n][:3,3] for n in thumb_links]
                directions=np.array([unit(positions[j+1]-positions[j]) for j in range(3)])
                return (directions-target_dirs).ravel()
            sol=least_squares(thumb_residual,[.45,.9,.5],
                              bounds=([bounds[n][0] for n in thumb_names],
                                      [bounds[n][1] for n in thumb_names]),max_nfev=160)
            q.update(dict(zip(thumb_names,sol.x)))
            measurements[finger]['robot_bone_direction_fit_residual']=float(np.linalg.norm(sol.fun))
        else:
            q[f'right_hand_{finger}_joint1'] = mcp
            q[f'right_hand_{finger}_joint2'] = pip + .5*dip
    # Index ab/adduction is small and is not confidently recoverable under
    # tool occlusion; center it instead of copying a noisy 2-D lateral angle.
    q['right_hand_index_bend_joint'] = 0.
    raw = {n:float(q[n]) for n in k.hand}
    bounded = {n:float(np.clip(q[n], *bounds[n])) for n in k.hand}
    return bounded, {'landmark_field':'pixel_xy_joint_angle_proxies' if use_2d else source,
                     'world_3d_used_for_strong_supervision':not use_2d,
                     'canonicalization':canonical,
                     'measured_finger_angles':measurements,
                     'raw_robot_joint_prior':raw,
                     'canonical_landmarks':p.tolist()}


def read_evidence(path, k, bounds, time_start=None, time_end=None):
    payload = json.loads(Path(path).read_text())
    frames = payload.get('frames', [])
    selected = []
    for f in frames:
        t = float(f.get('time_s', 0))
        if time_start is not None and t < time_start: continue
        if time_end is not None and t > time_end: continue
        if f.get('usable_for_retargeting', True) is False: continue
        if f.get('hand_role') not in (None,'tweezer_hand','forceps_hand'): continue
        if f.get('landmarks_world_m') is None and f.get('landmarks_normalized') is None: continue
        try: q, info = landmark_joint_prior(f, k, bounds)
        except (ValueError, TypeError): continue
        selected.append((f, q, info))
    if not selected:
        raise ValueError('No usable measured human landmark frames; refusing to invent a video prior')
    prior = {n:float(np.median([item[1][n] for item in selected])) for n in k.hand}
    std = {n:float(np.std([item[1][n] for item in selected])) for n in k.hand}
    evidence = {
        'landmarks_path':str(Path(path).resolve()),
        'landmarks_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'source_url':payload.get('source_url'), 'video_sha256':payload.get('video_sha256'),
        'selected_time_range_s':[min(float(f[0].get('time_s',0)) for f in selected),
                                 max(float(f[0].get('time_s',0)) for f in selected)],
        'selected_frame_count':len(selected),
        'selected_frame_indices':[f[0].get('frame_index') for f in selected],
        'estimator_methods':list({str(f[0].get('method','unspecified')) for f in selected}),
        'confidence_note':'A MediaPipe handedness score is not landmark-location accuracy. Occluded points remain model estimates.',
        'frame_diagnostics':[{'time_s':f.get('time_s'),'handedness':f.get('handedness'),
                              'frame_path':f.get('frame_path'),'annotated_frame_path':f.get('annotated_frame_path'),
                              'visual_review':f.get('visual_review'),'hand_role':f.get('hand_role'),
                              'confidence':f.get('confidence'),'occlusion_notes':f.get('occlusion_notes'),
                              **info} for f,_,info in selected],
    }
    return prior, std, evidence


def mesh_axis_surface(path, axis, fixed_axes, fixed_values, maximum=True):
    """Intersect an axis-aligned line with source STL triangles exactly."""
    tr=stl_triangles(path);A=tr[:,0,list(fixed_axes)];B=tr[:,1,list(fixed_axes)];C=tr[:,2,list(fixed_axes)]
    U,V,P=B-A,C-A,np.asarray(fixed_values)-A
    det=U[:,0]*V[:,1]-U[:,1]*V[:,0];good=np.abs(det)>1e-12
    u=np.zeros(len(tr));v=np.zeros(len(tr))
    u[good]=(P[good,0]*V[good,1]-P[good,1]*V[good,0])/det[good]
    v[good]=(U[good,0]*P[good,1]-U[good,1]*P[good,0])/det[good]
    good&=(u>=0)&(v>=0)&(u+v<=1)
    points=tr[:,0]+u[:,None]*(tr[:,1]-tr[:,0])+v[:,None]*(tr[:,2]-tr[:,0])
    indices=np.flatnonzero(good)
    if not len(indices):raise ValueError('Requested surface ray missed the source mesh')
    hit=indices[np.argmax(points[indices,axis]) if maximum else np.argmin(points[indices,axis])]
    normal=unit(np.cross(tr[hit,1]-tr[hit,0],tr[hit,2]-tr[hit,0]))
    return points[hit],normal


def fit_precision_contact(k, prior, bounds, pinch_z=.05, axial_offset=None):
    """Fit a thumb-pad / lateral-index precision grasp inside comfort bounds.

    The middle finger starts relaxed and is available to the learned policy as
    a support finger. Its initial contact is measured, never asserted here.
    """
    names = k.hand[:6]
    lo = np.array([bounds[n][0] for n in names]); hi = np.array([bounds[n][1] for n in names])
    # Reserve index adduction travel for subsequent squeezing.
    lo[3], hi[3] = -.035, .035
    prior_vec = np.array([prior[n] for n in names])
    rng = np.random.default_rng(197)
    candidates = []
    for thumb_x in [.032, .038, .044]:
        for index_z,index_link,index_axial_offset in [
            (-.030,'right_hand_index_rota_link2',.000),
            (-.030,'right_hand_index_rota_link2',.010),
            (-.030,'right_hand_index_rota_link2',.015),
            (-.030,'right_hand_index_rota_link2',.020),
            (-.030,'right_hand_index_rota_link2',.025),
            (-.030,'right_hand_index_rota_link2',.035),
            (-.020,'right_hand_index_rota_link2',.020),
            (-.040,'right_hand_index_rota_link1',.0),
        ]:
            if axial_offset is not None and abs(index_axial_offset-axial_offset)>1e-8:continue
            pt,nt0=mesh_axis_surface(ASSET/'sources/spider/assets/right_hand_thumb_rota_link2.STL',2,(0,1),(thumb_x,0),False)
            pi,ni0=mesh_axis_surface(ASSET/'sources/spider/assets'/f'{index_link}.STL',0,(1,2),(0,index_z),True)
            half_thumb=.0045+pinch_z*np.sin(.05)+.0005*np.cos(.05)
            half_index=.0045+(pinch_z+index_axial_offset)*np.sin(.05)+.0005*np.cos(.05)
            gap=half_thumb+half_index-.0002
            def calc(q):
                Ts = k.fk(dict(zip(names,q)), root='right_hand_link')
                a,b = Ts['right_hand_thumb_rota_link2'], Ts[index_link]
                return (a[:3,:3]@pt+a[:3,3], b[:3,:3]@pi+b[:3,3],
                        a[:3,:3]@nt0, b[:3,:3]@ni0)
            def residual(q):
                t,i,nt,ni = calc(q)
                x=unit(nt-ni)
                forward=np.array([0.,.10,-1.]);z=unit(forward-x*(forward@x))
                return np.r_[(i-t-x*gap-z*index_axial_offset)*500., .6*(nt+ni),
                             .4*nt[2], (q-prior_vec)*np.array([.10,.05,.05,.04,.45,.20])]
            for restart in range(5):
                seed=np.clip(prior_vec if restart==0 else rng.uniform(lo,hi),lo+1e-7,hi-1e-7)
                sol=least_squares(residual,seed,bounds=(lo,hi),max_nfev=240)
                t,i,nt,ni=calc(sol.x); X=unit(nt-ni); D=np.array([0.,.10,-1.]); Z=unit(D-X*(D@X));Y=unit(np.cross(Z,X))
                # Different axial contact locations have different rest widths.
                # Keep the tool centered on both physical outer strip faces.
                c=(t+i-index_axial_offset*Z)/2+.5*(half_thumb-half_index)*X
                T=transform(c-pinch_z*Z,np.column_stack([X,Y,Z]))
                q={**prior,**dict(zip(names,map(float,sol.x)))}
                # Curl differences among unused fingers are retained from the
                # human median, but prevent fully straight or tightly fisted
                # noisy occlusion estimates from dominating initialization.
                for n in k.hand:
                    if '_ring_' in n or '_pinky_' in n:
                        relaxed=.45 if 'joint1' in n else .60
                        q[n]=float(np.clip(.60*q[n]+.40*relaxed,.25,.85))
                    elif '_mid_' in n:
                        relaxed=.60 if 'joint1' in n else .70
                        q[n]=float(np.clip(.60*q[n]+.40*relaxed,.25,1.10))
                candidates.append({'loss':float(np.linalg.norm(residual(sol.x))),
                                   'q':q,'T':T,'thumb_local':pt,'index_local':pi,
                                   'index_link':index_link,'index_axial_offset_m':index_axial_offset,
                                   'thumb_normal_local':nt0,'index_normal_local':ni0,
                                   'contact_separation_3d_m':float(np.linalg.norm(t-i)),
                                   'closing_axis_contact_separation_m':float((i-t)@X),
                                   'contact_separation_target_m':gap,
                                   'opposed_normal_dot':float(nt@ni),'direction_forward_cosine':float(-Z[2])})
    candidates.sort(key=lambda c:c['loss'])
    return candidates


def geometry_audit(k, q, T, hinge_offset=.0045, open_angle=.05, width=.007, index_link='right_hand_index_rota_link1'):
    """Exact STL-vertex penetration/clearance sampling against the two beams.

    This checks all sampled source vertices, not just tip marker locations.
    Triangle-surface intersection and simulator hull validation remain separate.
    """
    Ts=k.fk(q,root='right_hand_link');details=[]
    intended={'right_hand_thumb_rota_link2',index_link}
    for link in k.root.findall('link'):
        name=link.get('name');mesh=link.find('visual/geometry/mesh')
        if name not in Ts or mesh is None:continue
        v=np.unique(stl_triangles(ASSET/mesh.get('filename')).reshape(-1,3),axis=0)
        LT=np.linalg.inv(T)@Ts[name];v=v@LT[:3,:3].T+LT[:3,3]
        for sign in [-1,1]:
            R=Rotation.from_rotvec([0,sign*open_angle,0]).as_matrix()
            p=(v-np.array([sign*hinge_offset,0,0]))@R
            margins=np.stack([.0005-np.abs(p[:,0]),width/2-np.abs(p[:,1]),p[:,2],.14-p[:,2]],axis=1)
            inside=np.all(margins>0,axis=1)
            if inside.any():
                details.append({'link':name,'arm_sign':sign,'vertices_inside':int(inside.sum()),
                                'max_sampled_penetration_m':float(np.min(margins[inside],axis=1).max()),
                                'tool_z_range_m':[float(p[inside,2].min()),float(p[inside,2].max())],
                                'intended_contact_link':name in intended})
    return {'method':'Source STL vertices in exact finite beam boxes; not a continuous surface certification',
            'beam_width_m':width,'penetrations':details,
            'no_noncontact_vertex_penetrations':all(d['intended_contact_link'] for d in details),
            'initial_middle_contact_claimed':False}


def closure_reference(k, fit, bounds, reduction=.0035):
    names=k.hand[:6];q0=np.array([fit['q'][n] for n in names]);T=fit['T'];X=T[:3,0]
    local=[fit['thumb_local'],fit['index_local']]
    links=['right_hand_thumb_rota_link2',fit['index_link']]
    def pts(q):
        Ts=k.fk(dict(zip(names,q)),root='right_hand_link')
        return [Ts[n][:3,:3]@p+Ts[n][:3,3] for n,p in zip(links,local)]
    start=pts(q0);targets=[start[0]+reduction/2*X,start[1]-reduction/2*X]
    def residual(q):
        p=pts(q)
        return np.r_[(p[0]-targets[0])*200,(p[1]-targets[1])*200,(q-q0)*.02]
    sol=least_squares(residual,q0,bounds=([bounds[n][0] for n in names],[bounds[n][1] for n in names]),max_nfev=400)
    closed={**fit['q'],**dict(zip(names,map(float,sol.x)))}
    delta={n:float(closed[n]-fit['q'][n]) for n in k.hand}
    end=pts(sol.x)
    return closed,delta,{'requested_reduction_m':reduction,
                         'achieved_projected_reduction_m':float((start[1]-start[0]-end[1]+end[0])@X),
                         'contact_fit_errors_m':[float(np.linalg.norm(p-t)) for p,t in zip(end,targets)]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--landmarks',required=True,type=Path)
    p.add_argument('--output',default=ROOT/'configs/human_grasp.json',type=Path)
    p.add_argument('--time-start',type=float);p.add_argument('--time-end',type=float)
    p.add_argument('--pinch-z',default=.05,type=float)
    p.add_argument('--index-axial-offset',type=float,default=None)
    p.add_argument('--contact-mode',choices=['index_distal','index_proximal','auto'],default='index_distal')
    a=p.parse_args();k=Kinematics();bounds=natural_soft_bounds(k)
    prior,std,evidence=read_evidence(a.landmarks,k,bounds,a.time_start,a.time_end)
    candidates=fit_precision_contact(k,prior,bounds,a.pinch_z,a.index_axial_offset)
    accepted=[]
    for c in candidates:
        if a.contact_mode=='index_distal' and c['index_link']!='right_hand_index_rota_link2':continue
        if a.contact_mode=='index_proximal' and c['index_link']!='right_hand_index_rota_link1':continue
        audit=geometry_audit(k,c['q'],c['T'],index_link=c['index_link'])
        if audit['no_noncontact_vertex_penetrations'] and c['direction_forward_cosine']>.78:
            accepted.append((c,audit))
    if not accepted: raise RuntimeError('No natural pose passed the finite-beam vertex audit')
    fit,audit=accepted[0];closed,delta,closure=closure_reference(k,fit,bounds)
    result={'schema_version':2,'method':'measured human landmark prior + bounded geometric contact refinement',
            'q_reference':fit['q'],'closure_delta':delta,'q_open':fit['q'],'q_closed':closed,
            'T_hand_tool':fit['T'].tolist(),'tool_pinch_z':a.pinch_z,
            'soft_bounds':bounds,'human_joint_prior':prior,'human_joint_prior_std':std,
            'prior_weights':{n:(1.5 if '_ring_' in n or '_pinky_' in n else .6) for n in k.hand},
            'joint_refinement_delta':{n:fit['q'][n]-prior[n] for n in k.hand},
            'human_evidence':evidence,'geometry_validation':audit,'closure_validation':closure,
            'occluded_finger_regularization':{'middle':'0.6 measured prior + 0.4 hand-authored relaxed prior (.6,.7)',
                                               'ring_pinky':'0.6 measured prior + 0.4 hand-authored relaxed prior (.45,.6), clipped [.25,.85]',
                                               'reason':'These fingers are occluded in the selected video; their exact 3-D pose is not reliably observed.'},
            'grip_index_link':fit['index_link'],
            'contact_targets':{'thumb':{'link':'right_hand_thumb_rota_link2','point_local':fit['thumb_local'].tolist()},
                               'index':{'link':fit['index_link'],'point_local':fit['index_local'].tolist(),
                                        'tool_longitudinal_offset_from_thumb_m':fit['index_axial_offset_m']},
                               'middle':{'role':'relaxed support candidate for RL; no initial contact asserted'}},
            'tool_geometry':{'length_m':.14,'hinge_offset_m':.0045,'open_angle_rad':.05,'strip_thickness_m':.001,'strip_width_m':.007},
            'reference_quality':{'contact_loss':fit['loss'],
                                 'contact_separation_3d_m':fit['contact_separation_3d_m'],
                                 'closing_axis_contact_separation_m':fit['closing_axis_contact_separation_m'],
                                 'closing_axis_contact_separation_target_m':fit['contact_separation_target_m'],
                                 'opposed_normal_dot':fit['opposed_normal_dot'],'tool_axis_hand_forward_cosine':fit['direction_forward_cosine']},
            'limitations':['Monocular landmark estimates and occluded fingers are uncertain; this is not motion-capture ground truth.',
                           'The robot lacks human finger abduction and has two flexion joints per long finger. Geometric refinement is not claimed to reproduce every human landmark.',
                           'Two initial contact targets are verified geometrically; a three-point support must be established and measured in dynamics.',
                           'Finite-beam vertex audit is not a full triangle or PhysX convex-hull collision proof.']}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({'output':str(a.output),'frames':evidence['selected_frame_count'],
                       'quality':result['reference_quality'],'closure':closure,'geometry':audit},indent=2))


if __name__=='__main__':main()
