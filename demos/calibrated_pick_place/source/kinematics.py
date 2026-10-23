"""Minimal URDF kinematics for UR5e with a configured dexterous hand.

No dynamics or attachment assumptions. Coordinates in meters and radians.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

ASSET = Path(__file__).resolve().parents[2] / 'assets/ur5e_xhand_right'

def xyz(text):
    return np.fromstring(text, sep=' ') if text else np.zeros(3)

def transform(translation=None, rotation=None):
    T = np.eye(4)
    if translation is not None:
        T[:3, 3] = translation
    if rotation is not None:
        T[:3, :3] = rotation
    return T

class Kinematics:
    def __init__(self, path=ASSET/'robot.urdf', *, palm_link='right_hand_link',
                 hand_root_link=None, arm_joint_names=None):
        self.path = Path(path).resolve()
        self.asset_dir = self.path.parent
        self.palm_link = palm_link
        self.hand_root_link = hand_root_link or palm_link
        self.root = ET.parse(self.path).getroot()
        self.joints = []
        self.by_name = {}
        for e in self.root.findall('joint'):
            o = e.find('origin')
            lim = e.find('limit')
            ax = e.find('axis')
            j = dict(name=e.get('name'), kind=e.get('type'),
                     parent=e.find('parent').get('link'), child=e.find('child').get('link'),
                     T=transform(xyz(o.get('xyz') if o is not None else None),
                                 Rotation.from_euler('xyz',xyz(o.get('rpy') if o is not None else None)).as_matrix()),
                     axis=xyz(ax.get('xyz')) if ax is not None else np.array([0,0,1]),
                     lower=float(lim.get('lower', -np.pi)) if lim is not None else 0.,
                     upper=float(lim.get('upper', np.pi)) if lim is not None else 0.)
            self.joints.append(j)
            self.by_name[j['name']] = j
        self.active = [j['name'] for j in self.joints if j['kind'] != 'fixed']
        self.arm = list(arm_joint_names) if arm_joint_names is not None else self.active[:6]
        if len(self.arm) != 6 or not set(self.arm).issubset(self.active):
            raise ValueError('The scene requires six named UR5e arm joints')
        self.hand = [n for n in self.active if n not in self.arm]
        self.hand_links = {self.hand_root_link}
        for _ in self.joints:
            self.hand_links.update(j['child'] for j in self.joints if j['parent'] in self.hand_links)

    def resolve_mesh(self, filename):
        """Resolve URDF mesh paths without depending on a global robot asset."""
        if filename.startswith('file://'):
            return Path(filename[7:])
        if filename.startswith('package://'):
            package_path = filename[len('package://'):]
            package, relative = package_path.split('/', 1)
            candidates = [self.asset_dir / relative, self.asset_dir.parent / package_path,
                          self.asset_dir / package_path]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(f'Cannot resolve package mesh {filename} from {self.asset_dir}')
        return self.asset_dir / filename

    def fk(self, q=None, root='base_link'):
        q = q or {}
        Ts = {root:np.eye(4)}
        pending = list(self.joints)
        while pending:
            progress = False
            for j in pending[:]:
                if j['parent'] not in Ts:
                    continue
                T = j['T']
                if j['kind'] in ('revolute', 'continuous'):
                    T = T @ transform(rotation=Rotation.from_rotvec(j['axis']*q.get(j['name'],0.)).as_matrix())
                elif j['kind'] == 'prismatic':
                    T = T @ transform(translation=j['axis']*q.get(j['name'],0.))
                elif j['kind'] != 'fixed':
                    raise ValueError(f"Unsupported joint type: {j['kind']}")
                Ts[j['child']] = Ts[j['parent']]@T
                pending.remove(j)
                progress = True
            if not progress:
                break
        return Ts

    def inverse_arm(self, target, initial=None, frame=None):
        frame = frame or self.palm_link
        q0 = np.asarray(initial if initial is not None else [0,-1.3,1.8,-2.1,-1.57,0.])
        def fun(q):
            T = self.fk(dict(zip(self.arm,q)))[frame]
            return np.r_[4*(T[:3,3]-target[:3,3]),Rotation.from_matrix(target[:3,:3].T@T[:3,:3]).as_rotvec()]
        sol=least_squares(fun,q0,bounds=([-6.28,-6.28,-3.14,-6.28,-6.28,-6.28],[6.28,6.28,3.14,6.28,6.28,6.28]),xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=500)
        return dict(q=dict(zip(self.arm,sol.x)),error=float(np.linalg.norm(fun(sol.x))),success=bool(sol.success))

def stl_triangles(path):
    data=Path(path).read_bytes()
    dt=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')])
    count = int.from_bytes(data[80:84], 'little') if len(data) >= 84 else -1
    if len(data) == 84 + count*50:
        return np.frombuffer(data[84:],dtype=dt)['vertices'].astype(float)
    vertices = [list(map(float, line.split()[1:])) for line in data.decode('ascii').splitlines()
                if line.strip().startswith('vertex ')]
    return np.asarray(vertices, dtype=float).reshape(-1,3,3)

if __name__ == '__main__':
    k=Kinematics()
    print('Active joints:',k.active)
    for name,T in k.fk(root='right_hand_link').items():
        if 'tip' in name:
            print(name,T[:3,3])
