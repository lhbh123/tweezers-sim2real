"""Select a UR5e IK branch with source-mesh clearance above the table.

This is reset-time planning only. It does not alter hand pose or simulation
collision filtering. Clearances use the supplied URDF collision mesh vertices.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from kinematics import stl_triangles,transform,xyz


class ArmTableClearance:
    def __init__(self,kin):
        self.kin=kin;self.vertices={};self.hand_vertices={}
        for link in kin.root.findall('link'):
            name=link.get('name')
            is_arm=name in ['shoulder_link','upper_arm_link','forearm_link','wrist_1_link','wrist_2_link','wrist_3_link']
            is_hand=name in kin.hand_links
            if not is_arm and not is_hand:continue
            pieces=[]
            for c in link.findall('collision'):
                m=c.find('geometry/mesh');o=c.find('origin')
                if m is None:continue
                path=kin.resolve_mesh(m.get('filename'))
                if path.suffix.lower()=='.stl':
                    v=np.unique(stl_triangles(path).reshape(-1,3),axis=0)
                elif path.suffix.lower()=='.obj':
                    v=np.asarray([list(map(float,s.split()[1:4])) for s in path.read_text().splitlines() if s.startswith('v ')])
                else:raise ValueError(f'Unsupported collision source mesh: {path}')
                v*=xyz(m.get('scale')) if m.get('scale') else np.ones(3)
                T=transform(xyz(o.get('xyz') if o is not None else None),Rotation.from_euler('xyz',xyz(o.get('rpy') if o is not None else None)).as_matrix())
                pieces.append(v@T[:3,:3].T+T[:3,3])
            if pieces:(self.vertices if is_arm else self.hand_vertices)[name]=np.concatenate(pieces)

    def clearance(self,q):
        fk=self.kin.fk(dict(zip(self.kin.arm,q)))
        return {n:float((v@fk[n][:3,:3].T+fk[n][:3,3])[:,2].min()) for n,v in self.vertices.items()}

    def hand_clearance(self,wrist_target,hand_joint_positions):
        relative=self.kin.fk(hand_joint_positions,root='wrist_3_link')
        result={}
        for name,vertices in self.hand_vertices.items():
            T=wrist_target@relative[name]
            result[name]=float((vertices@T[:3,:3].T+T[:3,3])[:,2].min())
        return result

    def inverse(self,target,initial,min_clearance=.015,preferred_seed=None,hand_joint_positions=None,min_hand_clearance=.001):
        hand_clearance=None
        if hand_joint_positions is not None:
            hand_clearance=self.hand_clearance(target,hand_joint_positions)
            failures={n:z for n,z in hand_clearance.items() if z<min_hand_clearance}
            if failures:
                raise ValueError(f'Hand source meshes lack table clearance at requested wrist pose: {failures}. Repose nonoperating fingers or adjust tool orientation before task reset.')
        initial=np.asarray(initial,float);seeds=[initial]
        if preferred_seed is not None:seeds.insert(0,np.asarray(preferred_seed,float))
        for elbow in [1.5,-1.5,2.5,-2.5]:
            for wrist in [-1.57,1.57]:seeds.append(np.array([initial[0],-1.3,elbow,-1.57,wrist,initial[-1]]))
        choices=[]
        for seed in seeds:
            result=self.kin.inverse_arm(target,seed,frame='wrist_3_link')
            if not result['success'] or result['error']>1e-4:continue
            q=np.array([result['q'][n] for n in self.kin.arm]);clearance=self.clearance(q)
            if min(clearance.values())<min_clearance:continue
            # Prefer a physically feasible nearby branch, keeping room for the approach.
            distance=float(np.linalg.norm((q-initial+np.pi)%(2*np.pi)-np.pi))
            result['table_clearance_by_link_m']=clearance
            if hand_clearance is not None:result['hand_table_clearance_by_link_m']=hand_clearance
            choices.append((distance,result))
        if not choices:raise ValueError('No source-mesh-clear IK branch found above the table')
        return min(choices,key=lambda x:x[0])[1]
