"""Cartesian increments applied to URDF joint COMMANDS, without erasing preload.

Only computes joint position targets. This is not a tool attachment, object-state
controller, trajectory schedule, or inverse dynamics model.
"""
import torch


class ArmCommandKinematics:
    def __init__(self, kin, device, dtype=torch.float32):
        self.device, self.dtype = device, dtype
        by_child = {j['child']: j for j in kin.joints}
        chain, link = [], 'wrist_3_link'
        while link != 'base_link':
            j = by_child[link]
            chain.append(j)
            link = j['parent']
        self.chain = []
        for j in reversed(chain):
            index = None if j['kind'] == 'fixed' else kin.arm.index(j['name'])
            self.chain.append((torch.as_tensor(j['T'], device=device, dtype=dtype),
                               torch.as_tensor(j['axis'], device=device, dtype=dtype), index))
        self.lo = torch.tensor([kin.by_name[n]['lower'] for n in kin.arm], device=device, dtype=dtype)
        self.hi = torch.tensor([kin.by_name[n]['upper'] for n in kin.arm], device=device, dtype=dtype)

    def fk_jacobian(self, q):
        n = len(q)
        T = torch.eye(4, device=q.device, dtype=q.dtype).repeat(n, 1, 1)
        origins, axes = [], []
        for origin, axis, index in self.chain:
            T = T @ origin
            if index is None:
                continue
            origins.append(T[:, :3, 3].clone())
            axes.append(T[:, :3, :3] @ axis)
            # Rodrigues formula for the actual URDF axis; no hard-coded DH signs.
            x, y, z = axis.unbind()
            zero = axis.new_zeros(())
            K = torch.stack((zero,-z,y,z,zero,-x,-y,x,zero)).reshape(3,3)
            angle = q[:, index, None, None]
            R = torch.eye(3, device=q.device, dtype=q.dtype) + angle.sin()*K + (1-angle.cos())*(K@K)
            motion = torch.eye(4, device=q.device, dtype=q.dtype).repeat(n,1,1)
            motion[:, :3, :3] = R
            T = T @ motion
        p = T[:, :3, 3]
        linear = torch.stack([torch.cross(a, p-o, dim=-1) for o,a in zip(origins,axes)], -1)
        angular = torch.stack(axes, -1)
        return T, torch.cat((linear, angular), 1)

    def increment(self, q, delta, reference_transform, workspace_lower, workspace_upper):
        current, _ = self.fk_jacobian(q)
        desired = (current[:, :3, 3] + delta).maximum(workspace_lower).minimum(workspace_upper)
        result = q.clone()
        identity = torch.eye(6, device=q.device, dtype=q.dtype).expand(len(q),6,6)
        for _ in range(2):
            T, J = self.fk_jacobian(result)
            # Spatial SO(3) error; small command increments make this local form appropriate.
            rotation_error = .5*sum(torch.cross(T[:,:3,i], reference_transform[:,:3,i],dim=-1) for i in range(3))
            error = torch.cat((desired-T[:,:3,3],rotation_error),-1)
            dq = (J.transpose(-1,-2) @ torch.linalg.solve(J@J.transpose(-1,-2)+.02**2*identity,error.unsqueeze(-1))).squeeze(-1)
            result = (result+dq).maximum(q-.025).minimum(q+.025).maximum(self.lo).minimum(self.hi)
        # A zero wrist command is an exact hold of the previous actuator targets.
        return torch.where((delta != 0).any(-1,keepdim=True), result, q)
