"""Measured tool-to-finger support, including friction and geometric web ROI.

Import and construct after Isaac has initialized. No simulation or state writes.
"""

class TripodContactReader:
    """Reusable GPU contact reader; construct after the scene has initialized.

    One view per tool body avoids the ambiguous 3N-source/N-filter mapping.
    ``read_support_tensors`` retains normal/friction forces on the GPU. Detailed
    point processing uses repeat_interleave and may synchronize its dynamic
    output allocation. with_web=False skips web ROI classification; middle
    underside contact classification remains available.
    Returned normal and friction vectors act on the tool source, not the finger.
    """
    groups = ['thumb', 'index', 'middle', 'wrist_palm', 'other']

    @staticmethod
    def classify(name):
        if 'thumb' in name: return 'thumb'
        if 'index' in name: return 'index'
        if 'mid' in name: return 'middle'
        if name == 'wrist_3_link': return 'wrist_palm'
        return 'other'

    def __init__(self, scene, capacity=32768, web_region=None, web_tool_body='Handle',
                 enable_face_metrics=False, face_config=None):
        import torch
        from pxr import Usd, UsdPhysics
        from isaacsim.core.prims import RigidPrim
        self.scene, self.web_region = scene, web_region
        self.enable_face_metrics = bool(enable_face_metrics)
        geo = scene.pose_config.get('tool_geometry', {})
        self.face_criteria = dict(width_margin_m=.002, axial_margin_m=.003,
            face_tolerance_m=.0004, min_inward_cos=.8660254,
            allow_tip_contact=False)
        aliases={'outer_plane_tolerance_m':'face_tolerance_m','inward_normal_cos_min':'min_inward_cos'}
        for config in [scene.pose_config.get('face_contact_criteria', {}), face_config or {}]:
            self.face_criteria.update({aliases.get(k,k):v for k,v in config.items()})
        self.face_config = self.face_criteria
        self.face_contact_point_mode = scene.pose_config.get('face_contact_point_mode', 'raw')
        if self.face_contact_point_mode not in ('raw', 'validated_tool_witness'):
            raise ValueError('face_contact_point_mode must be raw or validated_tool_witness')
        # This is a validation bound for a separation-derived witness, not an
        # enlargement of the existing broad-face acceptance tolerance.
        self.tool_witness_plane_tolerance_m = 20e-6
        self.face_geometry = dict(thickness_m=float(geo.get('strip_thickness_m', .001)),
            width_m=float(getattr(scene, 'tool_width', geo.get('strip_width_m', .007))),
            length_m=float(getattr(scene, 'tool_length', geo.get('length_m', .14))),
            tip_length_m=float(getattr(scene, 'tip_length', geo.get('tip_length_m', .015))))
        self.web_tool_body = web_tool_body.lower()
        if self.web_tool_body not in ['handle','left','right']:
            raise ValueError('web_tool_body must identify one actual tool rigid body')
        self.paths = [str(p.GetPath()) for p in Usd.PrimRange(scene.stage.GetPrimAtPath(scene.env_paths[0]+'/Robot'))
                      if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        self.names = [p.rsplit('/',1)[-1] for p in self.paths]
        self.views, self.row_env = [], []
        for body in ['Handle','Left','Right']:
            view = RigidPrim('/World/envs/env_.*/Tweezers/'+body, name='tripod_contacts_'+body,
                track_contact_forces=True, reset_xform_properties=False,
                contact_filter_prim_paths_expr=[p.replace('/env_0/','/env_.*/') for p in self.paths],
                max_contact_count=capacity)
            view.initialize()
            if len(view.prim_paths) != scene.num_envs:
                raise RuntimeError('Tool view must contain one source per environment')
            self.views.append(view)
            self.row_env.append(torch.tensor([int(p.split('/env_')[1].split('/')[0]) for p in view.prim_paths],
                                            device=scene.device,dtype=torch.long))
        self.role = torch.tensor([self.groups.index(self.classify(n)) for n in self.names],
                                 device=scene.device,dtype=torch.long)

    def face_point_metrics(self, point_local, force_local, jaw_sign, normal_contact=True):
        """Pure tensor geometry for one or many contact points in a jaw frame.

        X is strip thickness, Y width, Z hinge-to-tip. Left sign=-1 and
        right sign=+1 identify the outward broad plane. Reported torque is
        the contact force moment about the passive hinge, positive for closing;
        it is not the solver's total hinge reaction. Friction cannot qualify as
        an inward normal contact.
        """
        import torch
        g, c = self.face_geometry, self.face_config
        z = point_local[..., 2]
        blade_length = g['length_m'] - g['tip_length_m']
        is_tip = z > blade_length
        width = torch.where(is_tip, torch.full_like(z, min(.005, g['width_m'])),
                            torch.full_like(z, g['width_m']))
        z0 = torch.where(is_tip, torch.full_like(z, blade_length), torch.zeros_like(z))
        z1 = torch.where(is_tip, torch.full_like(z, g['length_m']),
                         torch.full_like(z, blade_length))
        width_margin = .5 * width - point_local[..., 1].abs()
        axial_margin = torch.minimum(z-z0, z1-z)
        plane_error = (jaw_sign*point_local[..., 0] - .5*g['thickness_m']).abs()
        inward = -jaw_sign*force_local[..., 0]
        alignment = inward / force_local.norm(dim=-1).clamp_min(1e-12)
        geometric = ((width_margin >= c['width_margin_m']) &
                     (axial_margin >= c['axial_margin_m']) &
                     (plane_error <= c['face_tolerance_m']))
        if not c['allow_tip_contact']:
            geometric &= ~is_tip
        qualified = geometric & (alignment >= c['min_inward_cos']) & (inward > 1e-8)
        if not normal_contact:
            qualified = torch.zeros_like(qualified)
        torque = -jaw_sign * torch.cross(point_local, force_local, dim=-1)[..., 1]
        return {'point_jaw_local_m': point_local, 'force_jaw_local_N': force_local,
                'jaw_outer_face_sign': torch.full_like(z, jaw_sign),
                'jaw_face_is_tip': is_tip, 'broad_face_width_margin_m': width_margin,
                'broad_face_axial_margin_m': axial_margin,
                'broad_face_outer_plane_error_m': plane_error,
                'broad_face_inward_alignment_cos': alignment,
                'broad_face_geometric_candidate': geometric, 'broad_face_qualified': qualified,
                'inward_force_component_N': inward, 'closing_contact_torque_Nm': torque}

    def face_metric_definition(self):
        return {'axes':'jaw local X thickness, Y width, Z hinge-to-tip',
                'jaw_order':['left','right'], 'role_order':['thumb','index'],
                'outer_face_sign':[-1,1], 'closing_hinge_Y_sign':[1,-1],
                'geometry':self.face_geometry, 'criteria':self.face_config,
                'contact_point_mode':self.face_contact_point_mode,
                'contact_point_policy':{
                    'raw_point':'Original PhysX contact point is preserved in point_jaw_local_m and point_jaw_raw_local_m.',
                    'candidate':'point_jaw_raw_local_m - separation_m * normal_jaw_local',
                    'candidate_validation':'Normal contact, negative signed force scalar, finite unit normal, and candidate on known jaw outer plane within 20 um. Otherwise use raw point.',
                    'candidate_plane_tolerance_m':self.tool_witness_plane_tolerance_m,
                    'scope':'Empirically validated for current box-jaw/convex-finger contacts. Not a universal PhysX witness-point contract.',
                    'metric_point':'face_metric_point_jaw_local_m; width, axial, plane and alignment thresholds are unchanged.'},
                'force_scope':'Forces on the tool. broad_face_inward_force_N and broad_face_closing_torque_Nm use qualified NORMAL contacts only. Friction is reported separately and cannot satisfy those normal metrics.',
                'torque_scope':'Contact-force moment about jaw rigid origin/hinge; not full solver joint reaction.',
                'missing_contact_margin_value':0.}

    def face_contact_metrics(self, point_local, force_local, jaw_sign,
                             normal_contact=True, normal_local=None, separation=None):
        """Classify a raw point, optionally using a validated tool-side witness.

        PhysX reports a contact point and signed separation, but its tensor API
        does not guarantee the point is on the sensor body's surface. Replays
        of this project's box-jaw contacts establish p - separation * normal
        as a tool-side candidate for negative signed scalar forces. We only
        use that candidate if it independently lands on the known outer plane;
        there is no arbitrary projection or broader face tolerance here.
        """
        import torch
        candidate = point_local
        shape = point_local.shape[:-1]
        finite = torch.isfinite(point_local).all(-1) & torch.isfinite(force_local).all(-1)
        valid = torch.zeros(shape, dtype=torch.bool, device=point_local.device)
        scalar = torch.zeros(shape, dtype=point_local.dtype, device=point_local.device)
        if normal_contact and normal_local is not None and separation is not None:
            separation = separation.reshape(shape)
            candidate = point_local - separation[..., None] * normal_local
            scalar = (force_local * normal_local).sum(-1)
            finite = (finite & torch.isfinite(normal_local).all(-1) &
                      torch.isfinite(separation) & torch.isfinite(candidate).all(-1))
            plane_error = (jaw_sign*candidate[..., 0] - .5*self.face_geometry['thickness_m']).abs()
            valid = (finite & (scalar < -1e-8) &
                     ((normal_local.norm(dim=-1)-1.).abs() <= 1e-3) &
                     (plane_error <= self.tool_witness_plane_tolerance_m))
        else:
            plane_error = (jaw_sign*candidate[..., 0] - .5*self.face_geometry['thickness_m']).abs()
        used = valid & (self.face_contact_point_mode == 'validated_tool_witness')
        metric_point = torch.where(used[..., None], candidate, point_local)
        metrics = self.face_point_metrics(metric_point, force_local, jaw_sign, normal_contact)
        raw_torque = -jaw_sign * torch.cross(point_local, force_local, dim=-1)[..., 1]
        candidate_torque = -jaw_sign * torch.cross(candidate, force_local, dim=-1)[..., 1]
        metrics.update({
            'point_jaw_local_m':point_local,
            'point_jaw_raw_local_m':point_local,
            'point_jaw_reconstructed_local_m':candidate,
            'face_metric_point_jaw_local_m':metric_point,
            'tool_surface_witness_valid':valid,
            'tool_surface_witness_used':used,
            'tool_surface_witness_plane_error_m':plane_error,
            'raw_outer_plane_error_m':(jaw_sign*point_local[..., 0] - .5*self.face_geometry['thickness_m']).abs(),
            'normal_force_signed_scalar_N':scalar,
            'raw_closing_contact_torque_Nm':raw_torque,
            'witness_closing_torque_difference_Nm':candidate_torque-raw_torque,
        })
        return metrics

    def read_raw(self, dt):
        self.last_raw = [{'matrix':v.get_contact_force_matrix(dt=dt),
                 'normal':v.get_contact_force_data(dt=dt),
                 'friction':v.get_friction_data(dt=dt)} for v in self.views]
        return self.last_raw

    def read_support_tensors(self, state, dt, with_web=True, with_faces=None):
        import torch
        from isaaclab.utils.math import quat_apply, quat_inv
        n, nf = self.scene.num_envs, len(self.names)
        normal = torch.zeros((n,3,nf,3),device=self.scene.device)
        friction = torch.zeros_like(normal)
        web_counts = torch.zeros(n,device=self.scene.device,dtype=torch.long)
        middle_under_counts = torch.zeros_like(web_counts)
        web_normal = torch.zeros((n,3),device=self.scene.device)
        web_friction = torch.zeros_like(web_normal)
        with_faces = self.enable_face_metrics if with_faces is None else bool(with_faces)
        face = {}
        if with_faces:
            for key in ['broad_face_inward_force_N','broad_face_contact_count',
                        'broad_face_closing_torque_Nm','broad_face_friction_inward_force_N',
                        'broad_face_friction_closing_torque_Nm','broad_face_all_normal_point_count',
                        'broad_face_all_normal_inward_force_N','broad_face_all_normal_closing_torque_Nm',
                        'broad_face_outer_plane_error_max_m','broad_face_surface_witness_valid_count',
                        'broad_face_surface_witness_used_count']:
                face[key] = torch.zeros((n,2,2),device=self.scene.device)
            for key in ['broad_face_width_margin_min_m','broad_face_axial_margin_min_m',
                        'broad_face_inward_alignment_min']:
                face[key] = torch.full((n,2,2),float('inf'),device=self.scene.device)
            for key in ['broad_face_normal_force_w','broad_face_friction_force_w']:
                face[key] = torch.zeros((n,2,2,3),device=self.scene.device)

        def accumulate_face(b, env_ids, columns, points, vectors, normal_contact,
                            normals=None, separations=None):
            if not with_faces or b == 0:
                return
            roles = self.role[columns]
            chosen = (roles < 2) & (vectors.norm(dim=-1)>1e-8)
            env_ids, roles, points, vectors = [x[chosen] for x in [env_ids,roles,points,vectors]]
            if normals is not None:normals=normals[chosen]
            if separations is not None:separations=separations[chosen]
            body = state[['handle_root_state_w','left_root_state_w','right_root_state_w'][b]][env_ids]
            inv = quat_inv(body[:,3:7])
            local = quat_apply(inv, points-body[:,:3])
            local_force = quat_apply(inv, vectors)
            local_normal = quat_apply(inv,normals) if normals is not None else None
            metrics = self.face_contact_metrics(local,local_force,-1 if b==1 else 1,
                normal_contact,local_normal,separations)
            flat = env_ids*4 + (b-1)*2 + roles
            def add(key, values):
                face[key].view(-1).scatter_add_(0,flat,values)
            if normal_contact:
                mask = metrics['broad_face_qualified'].to(vectors.dtype)
                add('broad_face_contact_count',mask)
                add('broad_face_inward_force_N',metrics['inward_force_component_N']*mask)
                add('broad_face_closing_torque_Nm',metrics['closing_contact_torque_Nm']*mask)
                add('broad_face_all_normal_point_count',torch.ones_like(mask))
                add('broad_face_all_normal_inward_force_N',metrics['inward_force_component_N'])
                add('broad_face_all_normal_closing_torque_Nm',metrics['closing_contact_torque_Nm'])
                add('broad_face_surface_witness_valid_count',metrics['tool_surface_witness_valid'].to(vectors.dtype))
                add('broad_face_surface_witness_used_count',metrics['tool_surface_witness_used'].to(vectors.dtype))
                for output,source in [('broad_face_width_margin_min_m','broad_face_width_margin_m'),
                        ('broad_face_axial_margin_min_m','broad_face_axial_margin_m'),
                        ('broad_face_inward_alignment_min','broad_face_inward_alignment_cos')]:
                    face[output].view(-1).scatter_reduce_(0,flat,metrics[source],reduce='amin',include_self=True)
                face['broad_face_outer_plane_error_max_m'].view(-1).scatter_reduce_(
                    0,flat,metrics['broad_face_outer_plane_error_m'],reduce='amax',include_self=True)
                force_key='broad_face_normal_force_w'
            else:
                mask=metrics['broad_face_geometric_candidate'].to(vectors.dtype)
                add('broad_face_friction_inward_force_N',metrics['inward_force_component_N']*mask)
                add('broad_face_friction_closing_torque_Nm',metrics['closing_contact_torque_Nm']*mask)
                force_key='broad_face_friction_force_w'
            face[force_key].view(-1,3).scatter_add_(0,flat[:,None].expand(-1,3),vectors*mask[:,None])
        for b,(raw,envs) in enumerate(zip(self.read_raw(dt),self.row_env)):
            normal[envs,b] = raw['matrix']
            ff,fp,counts,starts = raw['friction']
            prefix = torch.cat((torch.zeros_like(ff[:1]),torch.cumsum(ff,dim=0)),dim=0)
            friction[envs,b] = prefix[(starts+counts).long()]-prefix[starts.long()]
            f,p,norm,sep,counts,starts = raw['normal']
            counts, starts = counts.flatten().long(), starts.flatten().long()
            pairs = torch.arange(n*nf,device=self.scene.device).repeat_interleave(counts)
            offsets = torch.arange(len(pairs),device=self.scene.device) - (counts.cumsum(0)-counts).repeat_interleave(counts)
            point_ids = starts.repeat_interleave(counts)+offsets
            point_envs = envs[pairs//nf]
            accumulate_face(b,point_envs,pairs%nf,p[point_ids],f[point_ids]*norm[point_ids],True,
                norm[point_ids],sep[point_ids])
            body=state[['handle_root_state_w','left_root_state_w','right_root_state_w'][b]][point_envs]
            axis=quat_apply(body[:,3:7],torch.tensor([0.,0.,1.],device=self.scene.device).expand(len(body),3))
            delta=p[point_ids]-body[:,:3]
            radial_z=delta[:,2]-axis[:,2]*(delta*axis).sum(-1)
            under=(self.role[pairs%nf]==2)&(radial_z<-.0001)&(f[point_ids,0].abs()>1e-8)
            middle_under_counts.scatter_add_(0,point_envs,under.long())
            if with_web and self.web_region and b==['handle','left','right'].index(self.web_tool_body):
                hand = quat_apply(quat_inv(state['palm_quat_w'][point_envs]),p[point_ids]-state['palm_pos_w'][point_envs])
                region=self.web_region
                if 'bbox_min_m' in region:
                    low=torch.as_tensor(region['bbox_min_m'],device=self.scene.device)
                    high=torch.as_tensor(region['bbox_max_m'],device=self.scene.device)
                    inside=((hand>=low)&(hand<=high)).all(-1)
                else:
                    center=torch.as_tensor(region['center_m'],device=self.scene.device)
                    inside=(hand-center).norm(dim=-1)<=region['radius_m']
                inside &= (pairs%nf == self.names.index('wrist_3_link')) & (f[point_ids,0].abs()>1e-8)
                web_counts.scatter_add_(0,point_envs,inside.long())
                vectors=f[point_ids]*norm[point_ids]*inside[:,None]
                web_normal.scatter_add_(0,point_envs[:,None].expand(-1,3),vectors)
                # Friction anchors differ from normal contact points; classify
                # their own coordinates instead of reusing normal-buffer IDs.
                counts, starts = raw['friction'][2].flatten().long(), raw['friction'][3].flatten().long()
                pairs = torch.arange(n*nf,device=self.scene.device).repeat_interleave(counts)
                offsets = torch.arange(len(pairs),device=self.scene.device) - (counts.cumsum(0)-counts).repeat_interleave(counts)
                point_ids = starts.repeat_interleave(counts)+offsets
                point_envs = envs[pairs//nf]
                hand = quat_apply(quat_inv(state['palm_quat_w'][point_envs]),fp[point_ids]-state['palm_pos_w'][point_envs])
                if 'bbox_min_m' in region:
                    inside=((hand>=low)&(hand<=high)).all(-1)
                else:
                    inside=(hand-center).norm(dim=-1)<=region['radius_m']
                inside &= pairs%nf == self.names.index('wrist_3_link')
                web_friction.scatter_add_(0,point_envs[:,None].expand(-1,3),ff[point_ids]*inside[:,None])
            if with_faces and b > 0:
                counts, starts = raw['friction'][2].flatten().long(), raw['friction'][3].flatten().long()
                pairs = torch.arange(n*nf,device=self.scene.device).repeat_interleave(counts)
                offsets = torch.arange(len(pairs),device=self.scene.device) - (counts.cumsum(0)-counts).repeat_interleave(counts)
                point_ids = starts.repeat_interleave(counts)+offsets
                accumulate_face(b,envs[pairs//nf],pairs%nf,fp[point_ids],ff[point_ids],False)
        grouped_normal=torch.stack([normal[:,:,self.role==g].sum((1,2)) for g in range(len(self.groups))],dim=1)
        grouped_friction=torch.stack([friction[:,:,self.role==g].sum((1,2)) for g in range(len(self.groups))],dim=1)
        strength=torch.stack([normal[:,:,self.role==g].norm(dim=-1).sum((1,2)) for g in range(len(self.groups))],dim=1)
        if with_faces:
            for key in ['broad_face_width_margin_min_m','broad_face_axial_margin_min_m',
                        'broad_face_inward_alignment_min']:
                face[key] = torch.where(face['broad_face_all_normal_point_count']>0,face[key],torch.zeros_like(face[key]))
        return {'normal_force_w':normal,'friction_force_w':friction,
                'role_normal_force_w':grouped_normal,'role_friction_force_w':grouped_friction,
                'role_contact_strength_N':strength,'role_support_z_N':(grouped_normal+grouped_friction)[...,2],
                'web_contact_count':web_counts,'web_normal_force_w':web_normal,
                'web_friction_force_w':web_friction,'web_support_z_N':(web_normal+web_friction)[:,2],
                'middle_under_contact_count':middle_under_counts, **face}

    def audit_env_points(self, state, dt, env_id=0, refresh=False):
        """Small CPU record for one environment; call only at audit cadence.

        Uses the last tensor read unless refresh=True. Points retain source body,
        filter link, hand coordinates, actual force direction and ROI membership.
        """
        import torch
        from isaaclab.utils.math import quat_apply, quat_inv
        raw_views=self.read_raw(dt) if refresh or not hasattr(self,'last_raw') else self.last_raw
        result={'environment':int(env_id),'normal_contacts':[],'friction_anchors':[]}
        def classify_point(point):
            hand=quat_apply(quat_inv(state['palm_quat_w'][env_id]),point-state['palm_pos_w'][env_id])
            region=self.web_region
            inside=False
            if region:
                if 'bbox_min_m' in region:
                    inside=bool(((hand>=torch.as_tensor(region['bbox_min_m'],device=hand.device))&
                                 (hand<=torch.as_tensor(region['bbox_max_m'],device=hand.device))).all())
                else:
                    inside=bool((hand-torch.as_tensor(region['center_m'],device=hand.device)).norm()<=region['radius_m'])
            return hand.tolist(),inside
        def face_fields(b,body,point,force,normal_contact,normal=None,separation=None):
            if b==0:return {}
            inv=quat_inv(body[3:7])
            local=quat_apply(inv,point-body[:3]);local_force=quat_apply(inv,force)
            local_normal=quat_apply(inv,normal) if normal is not None else None
            metrics=self.face_contact_metrics(local,local_force,-1 if b==1 else 1,
                normal_contact,local_normal,separation)
            if normal_contact and normal is not None and separation is not None:
                metrics['point_tool_surface_candidate_world_m']=point-separation.reshape(())*normal
            return {k:v.tolist() for k,v in metrics.items()}
        for b,(raw,envs) in enumerate(zip(raw_views,self.row_env)):
            row=int((envs==env_id).nonzero(as_tuple=False)[0,0])
            f,p,norm,sep,count,index=raw['normal']
            body=state[['handle_root_state_w','left_root_state_w','right_root_state_w'][b]][env_id]
            axis=quat_apply(body[3:7],torch.tensor([0.,0.,1.],device=body.device))
            for col,name in enumerate(self.names):
                begin=int(index[row,col]);end=begin+int(count[row,col])
                for k in range(begin,end):
                    if float(f[k,0].abs())<1e-8:continue
                    hand,inside=classify_point(p[k]);delta=p[k]-body[:3]
                    result['normal_contacts'].append({'tool':['handle','left','right'][b],'robot':name,
                        'role':self.classify(name),'point_world':p[k].tolist(),'point_hand':hand,
                        'force_on_tool_N':(f[k,0]*norm[k]).tolist(),'normal_world':norm[k].tolist(),
                        'separation_m':float(sep[k,0]),'radial_offset_world_z_m':float(delta[2]-axis[2]*(delta*axis).sum()),
                        'within_web_region':name=='wrist_3_link' and inside and ['handle','left','right'][b]==self.web_tool_body,
                        **face_fields(b,body,p[k],f[k,0]*norm[k],True,norm[k],sep[k,0])})
                ff,fp,fc,fi=raw['friction'];begin=int(fi[row,col]);end=begin+int(fc[row,col])
                for k in range(begin,end):
                    if float(ff[k].norm())<1e-8:continue
                    hand,inside=classify_point(fp[k])
                    result['friction_anchors'].append({'tool':['handle','left','right'][b],'robot':name,
                        'role':self.classify(name),'point_world':fp[k].tolist(),'point_hand':hand,
                        'force_on_tool_N':ff[k].tolist(),'within_web_region':name=='wrist_3_link' and inside and ['handle','left','right'][b]==self.web_tool_body,
                        **face_fields(b,body,fp[k],ff[k],False)})
        return result

