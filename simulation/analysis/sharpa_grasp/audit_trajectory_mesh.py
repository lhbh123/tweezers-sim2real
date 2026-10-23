"""Sample saved W01 motion for source-triangle self/tool intersections, without Isaac.

Run: .venv-human/bin/python analysis/sharpa_grasp/audit_trajectory_mesh.py RUN
Only completed saved trajectories are read. No simulator state or task result is changed.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import fcl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from kinematics import Kinematics, transform, xyz


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pose_transform(state):
    return transform(state[:3], Rotation.from_quat(np.roll(state[3:7], -1)).as_matrix())


def corners(bounds):
    return np.asarray(list(itertools.product(*bounds.T)))


def aabb_distance(a, b):
    delta = np.maximum(0., np.maximum(a[..., 0, :] - b[..., 1, :], b[..., 0, :] - a[..., 1, :]))
    return np.linalg.norm(delta, axis=-1)


class SourceMeshAudit:
    def __init__(self, metadata, base_height=.72):
        self.meta = metadata
        self.base_height = base_height
        self.urdf = Path(metadata['robot_asset_dir']) / 'robot.urdf'
        self.kin = Kinematics(self.urdf, palm_link=metadata['palm_link'],
                              hand_root_link=metadata['hand_root_link'],
                              arm_joint_names=metadata['arm_joint_names'])
        self.names = metadata['joint_names']
        if set(self.names) != set(self.kin.active):
            raise ValueError('Trajectory metadata joint names do not match source URDF')
        if 'W01' not in metadata['robot_model']:
            raise ValueError('This audit is for the W01 hand')
        self.meshes, self.info, self.objects, self.bounds_corners = {}, {}, {}, {}
        for link in self.kin.root.findall('link'):
            name = link.get('name')
            if name not in self.kin.hand_links:
                continue
            cols = link.findall('collision')
            for i, col in enumerate(cols):
                key = name if len(cols) == 1 else f'{name}#{i}'
                element = col.find('geometry/mesh')
                if element is None:
                    raise ValueError(f'Unhandled non-mesh hand collider {key}')
                path = self.kin.resolve_mesh(element.get('filename')).resolve()
                mesh = trimesh.load_mesh(path, process=True)
                mesh.apply_scale(xyz(element.get('scale', '1 1 1')))
                o = col.find('origin')
                if o is not None:
                    mesh.apply_transform(transform(xyz(o.get('xyz')), Rotation.from_euler('xyz', xyz(o.get('rpy'))).as_matrix()))
                self.meshes[key] = mesh
                self.info[key] = dict(link=name, source=str(path), sha256=sha(path),
                                      triangles=len(mesh.faces), watertight=bool(mesh.is_watertight))
                self.bounds_corners[key] = corners(mesh.bounds)
                model = fcl.BVHModel()
                model.beginModel(len(mesh.vertices), len(mesh.faces))
                model.addSubModel(mesh.vertices, mesh.faces)
                model.endModel()
                self.objects[key] = fcl.CollisionObject(model)
        self.keys = list(self.meshes)
        self.parents = {j['child']: j['parent'] for j in self.kin.joints}
        self.kinds = {j['child']: j['kind'] for j in self.kin.joints}
        self.colliding_links = {v['link'] for v in self.info.values()}
        self.fixed = {name: self.fixed_root(name) for name in self.kin.hand_links}
        evidence_path = ROOT/'analysis/sharpa_grasp/official_usd_collision_filters.json'
        self.filter_source = dict(path=str(evidence_path), sha256=sha(evidence_path))
        self.official_pairs = set()
        for prim in json.loads(evidence_path.read_text()):
            for relation in prim['relationships']:
                if relation['name'] == 'physics:filteredPairs':
                    for target in relation['targets']:
                        self.official_pairs.add(frozenset([Path(prim['prim']).name, Path(target).name]))
        if len(self.official_pairs) != 7:
            raise ValueError('Expected the seven extracted official hand mechanism filters')
        self.self_pairs, self.excluded_pairs = [], []
        for a, b in itertools.combinations(self.keys, 2):
            why = self.mechanical(self.info[a]['link'], self.info[b]['link'])
            if why:
                self.excluded_pairs.append(dict(pair=[a,b], reason=why))
            else:
                self.self_pairs.append((a,b))
        self.tool_meshes, self.tool_body, self.tool_corners = {}, {}, {}
        geo = metadata['tool_geometry']
        length, tip = geo['length_m'], geo.get('tip_length_m', .015)
        thickness, width = geo['strip_thickness_m'], geo['strip_width_m']
        def box(key, body, dimensions, center, rotation=None):
            mesh = trimesh.creation.box(dimensions)
            mesh.apply_transform(transform(center, rotation))
            self.tool_meshes[key] = mesh
            self.tool_body[key] = body
            self.tool_corners[key] = corners(mesh.bounds)
            model = fcl.BVHModel()
            model.beginModel(len(mesh.vertices), len(mesh.faces))
            model.addSubModel(mesh.vertices, mesh.faces)
            model.endModel()
            self.objects[key] = fcl.CollisionObject(model)
        tail = geo.get('handle_length_m', .006)
        box('joined_tail', 'tool', [2*geo['hinge_offset_m']+thickness, geo.get('handle_width_m', width), tail], [0,0,-tail/2])
        if geo.get('handle_extension'):
            extension = geo['handle_extension']
            start, end = (np.asarray(extension[k]) for k in ['start_m','end_m'])
            z = (end-start)/np.linalg.norm(end-start)
            x = np.array([1.,0.,0.])-z*z[0]
            x /= np.linalg.norm(x)
            box('tail_extension', 'tool', [extension['width_x_m'],extension['width_y_m'],np.linalg.norm(end-start)], (start+end)/2, np.column_stack((x,np.cross(z,x),z)))
        for side in ['left','right']:
            box(side+'_blade', side, [thickness,width,length-tip], [0,0,(length-tip)/2])
            box(side+'_tip', side, [thickness,min(.005,width),tip], [0,0,length-tip/2])
        assignment = metadata.get('face_pinch_assignment') or {'thumb':'left','index':'right'}
        self.targets = {(f'right_{finger}_elastomer', side+'_blade') for finger,side in assignment.items()}
        self.tool_pairs = list(itertools.product(self.keys, self.tool_meshes))
        self.nontarget_pairs = [(a,b) for a,b in self.tool_pairs if (self.info[a]['link'],b) not in self.targets]

    def fixed_root(self, link):
        while self.kinds.get(link) == 'fixed' and link in self.parents:
            link = self.parents[link]
        return link

    def mechanical(self, a, b):
        if frozenset([a,b]) in self.official_pairs:
            return 'official_USD_filtered_pair'
        if self.fixed[a] == self.fixed[b]:
            return 'same_fixed_rigid_assembly'
        for child, ancestor in [(a,b),(b,a)]:
            virtual = []
            while child in self.parents:
                child = self.parents[child]
                if self.fixed_root(child) == self.fixed[ancestor]:
                    return 'URDF_adjacent_collision_assemblies' + (':virtual_bridge='+','.join(virtual) if virtual else '')
                if child in self.colliding_links:
                    break
                # Only collider-free links may bridge a kinematic mechanism here.
                virtual.append(child)
        return None

    def state(self, q_array, states, origin, update_objects=False):
        fk = self.kin.fk(dict(zip(self.names, q_array)))
        world_to_hand = np.linalg.inv(transform(np.asarray(origin)+[0,0,self.base_height]) @ fk[self.kin.hand_root_link])
        root_to_hand = np.linalg.inv(fk[self.kin.hand_root_link])
        poses = {k: root_to_hand@fk[self.info[k]['link']] for k in self.keys}
        poses.update({k: world_to_hand@pose_transform(states[self.tool_body[k]]) for k in self.tool_meshes})
        bounds = {}
        for key, T in poses.items():
            local = self.bounds_corners.get(key, self.tool_corners.get(key))
            points = local@T[:3,:3].T+T[:3,3]
            bounds[key] = np.stack([points.min(0), points.max(0)])
            if update_objects:
                self.objects[key].setTransform(fcl.Transform(T[:3,:3], T[:3,3]))
        return bounds

    @staticmethod
    def lower_bound(bounds, pairs):
        return np.asarray([aabb_distance(bounds[a],bounds[b]) for a,b in pairs])

    def examine(self, bounds, pairs):
        lower = self.lower_bound(bounds,pairs)
        min_distance = float('inf')
        nearest, hits = None, []
        for j in np.argsort(lower):
            a,b = pairs[j]
            if lower[j] > min_distance+1e-12:
                break
            if lower[j] <= 1e-12:
                result = fcl.CollisionResult()
                fcl.collide(self.objects[a],self.objects[b],fcl.CollisionRequest(num_max_contacts=32,enable_contact=True),result)
                if result.is_collision:
                    hits.append(dict(pair=[a,b], max_fcl_witness_depth_m=max((float(x.penetration_depth) for x in result.contacts),default=0.), witness_points_hand_root_m=[np.asarray(x.pos).tolist() for x in result.contacts[:3]]))
                    min_distance, nearest = 0., [a,b]
                    continue
            if min_distance > 0:
                result = fcl.DistanceResult()
                dist = max(0.,float(fcl.distance(self.objects[a],self.objects[b],fcl.DistanceRequest(enable_nearest_points=True),result)))
                if dist < min_distance:
                    min_distance,nearest = dist,[a,b]
        return dict(min_distance_m=min_distance, nearest_pair=nearest, intersections=hits)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--trajectory',type=Path,help='Override RUN/trajectory.npz')
    p.add_argument('--metadata',type=Path,help='Override RUN/task.json')
    p.add_argument('--out',type=Path,help='Default RUN/source_mesh_trajectory_audit.json')
    p.add_argument('--stride',type=int,default=5)
    p.add_argument('--nearest-candidates',type=int,default=12,help='Additional smallest AABB-distance active samples per environment and pair class')
    p.add_argument('--base-height',type=float,default=.72,help='World base Z above environment origin; scene default .72 m')
    a=p.parse_args()
    if a.stride<1 or a.nearest_candidates<0:
        p.error('Invalid sampling parameters')
    trajectory=a.trajectory or a.run/'trajectory.npz'
    metadata=a.metadata or a.run/'task.json'
    task=json.loads(metadata.read_text())
    if 'task' in task: task=task['task']
    audit=SourceMeshAudit(task,a.base_height)
    with np.load(trajectory,allow_pickle=False) as data:
        d={k:data[k] for k in ['q','tool','left','right','active','origins','dt','phase']}
    model_hashes=a.run/'robot_asset_sha256.json'
    model_match=None
    if model_hashes.exists():
        expected=json.loads(model_hashes.read_text()).get(str(audit.urdf.resolve()))
        model_match=sha(audit.urdf)==expected if expected else None
        if model_match is False:
            raise ValueError('The robot source URDF changed since the saved run')
    active=np.asarray(d['active'],dtype=bool)
    if d['q'].shape[:2]!=active.shape or d['q'].shape[-1]!=len(audit.names):
        raise ValueError('Malformed trajectory or metadata')
    selected={}
    screened=[]
    started=time.monotonic()
    for env in range(active.shape[1]):
        ids=np.flatnonzero(active[:,env])
        if not len(ids): continue
        candidates=set(ids[::a.stride].tolist()+[int(ids[-1])])
        changes=ids[1:][d['phase'][ids[1:],env]!=d['phase'][ids[:-1],env]]
        candidates.update(changes.tolist());candidates.update((changes-1).tolist())
        rows=[]
        for step in ids:
            bounds=audit.state(d['q'][step,env],{k:d[k][step,env] for k in ['tool','left','right']},d['origins'][env])
            rows.append((int(step),float(audit.lower_bound(bounds,audit.self_pairs).min()),float(audit.lower_bound(bounds,audit.nontarget_pairs).min())))
        values=np.asarray(rows)
        for col in [1,2]:
            candidates.update(values[np.argsort(values[:,col],kind='stable')[:a.nearest_candidates],0].astype(int).tolist())
        selected[env]=sorted(x for x in candidates if active[x,env])
        screened.append(dict(environment=env, active_samples=len(ids), selected_samples=len(selected[env]), min_self_AABB_lower_bound_m=float(values[:,1].min()), min_nontarget_tool_AABB_lower_bound_m=float(values[:,2].min())))
        print(f'SCREEN env={env} active={len(ids)} selected={len(selected[env])}',flush=True)
    samples=[]
    for env,ids in selected.items():
        for index,step in enumerate(ids):
            bounds=audit.state(d['q'][step,env],{k:d[k][step,env] for k in ['tool','left','right']},d['origins'][env],True)
            self_result=audit.examine(bounds,audit.self_pairs)
            tool_result=audit.examine(bounds,audit.tool_pairs)
            target=[h for h in tool_result['intersections'] if (audit.info[h['pair'][0]]['link'],h['pair'][1]) in audit.targets]
            nontarget=[h for h in tool_result['intersections'] if h not in target]
            # Target contact may have distance zero; compute non-target minimum separately.
            other=audit.examine(bounds,audit.nontarget_pairs)
            samples.append(dict(environment=env,control_step=step,time_s=(step+1)*float(d['dt']),phase=int(d['phase'][step,env]),self=self_result,
                                tool=dict(min_nontarget_distance_m=other['min_distance_m'],nearest_nontarget_pair=other['nearest_pair'],target_pad_intersections=target,nontarget_intersections=nontarget)))
            if index%50==0: print(f'EXACT env={env} sample={index+1}/{len(ids)} step={step}',flush=True)
    offending=[dict(environment=s['environment'],control_step=s['control_step'],self_pairs=[h['pair'] for h in s['self']['intersections']],nontarget_tool_pairs=[h['pair'] for h in s['tool']['nontarget_intersections']]) for s in samples if s['self']['intersections'] or s['tool']['nontarget_intersections']]
    report=dict(schema_version=1,run=str(a.run.resolve()),trajectory=dict(path=str(trajectory.resolve()),sha256=sha(trajectory)),metadata=dict(path=str(metadata.resolve()),sha256=sha(metadata)),source_urdf=dict(path=str(audit.urdf),sha256=sha(audit.urdf),matches_run_hash=model_match),audit_script_sha256=sha(__file__),
                method='Original URDF collision triangles at saved joint positions; independently measured handle/left/right root poses; FCL discrete triangle intersection and nonnegative surface distance. No simulation rerun.',
                frame=dict(audit_frame=audit.kin.hand_root_link,base_height_m=a.base_height,base_pose='environment origin + [0,0,base_height], identity rotation'),
                sampling=dict(stride=a.stride,additional_nearest_AABB_candidates_per_class=a.nearest_candidates,phase_boundaries=True,active_control_samples=int(active.sum()),exact_samples=len(samples),screened=screened),
                official_filter_source=audit.filter_source,official_filter_pairs=[sorted(x) for x in sorted(audit.official_pairs,key=lambda p:sorted(p))],excluded_self_pairs=audit.excluded_pairs,
                source_meshes=audit.info,tool_geometry=task['tool_geometry'],target_pad_pairs=sorted(audit.targets),
                summary=dict(sampled_nonmechanism_self_intersection_samples=sum(bool(s['self']['intersections']) for s in samples),sampled_nontarget_tool_intersection_samples=sum(bool(s['tool']['nontarget_intersections']) for s in samples),sampled_target_pad_intersection_samples=sum(bool(s['tool']['target_pad_intersections']) for s in samples),minimum_sampled_nonmechanism_self_surface_distance_m=min((s['self']['min_distance_m'] for s in samples),default=None),minimum_sampled_nontarget_tool_surface_distance_m=min((s['tool']['min_nontarget_distance_m'] for s in samples),default=None)),
                offending_samples=offending,samples=samples,wall_seconds=time.monotonic()-started,
                limitations=['Control-state sampling is not continuous collision detection or an all-physics-substep audit. AABB candidate selection supplies lower bounds only; exact reported minima cover sampled states only.','Adjacent source collision assemblies across collider-free virtual links and the exact seven official pairs are listed and excluded; no other cross-finger pairs are suppressed.','FCL triangle witness depth is not a reliable maximum volumetric penetration. Mesh intersection is not a solid-containment proof, especially for nonwatertight meshes.','Intended thumb/index elastomer versus assigned blade intersections remain separately reported; contact quality and broad-face force must come from physics.','URDF joint FK omits solver joint compliance. PhysX convex collision approximations may differ from the source triangles.','Arm, table and block checks belong to the separate full-control geometry validator. This report does not assert task success.'])
    output=a.out or a.run/'source_mesh_trajectory_audit.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),summary=report['summary'],exact_samples=len(samples),wall_seconds=report['wall_seconds']),indent=2),flush=True)


if __name__=='__main__': main()
