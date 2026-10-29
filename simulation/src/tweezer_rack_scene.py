"""Passive, upward-removable tail cradle for the existing physical tweezers."""
from pxr import Gf, Sdf, UsdShade
import numpy as np
from scipy.spatial.transform import Rotation
from rl_scene import BatchedTweezersScene


class TweezerRackScene(BatchedTweezersScene):
    def __init__(self, rack_root_xyz, rack_yaw_deg=0., rack_guides=True,
                 block_xy=(0., .65), **kwargs):
        self.rack_root_xyz = tuple(rack_root_xyz)
        self.rack_guides = bool(rack_guides)
        self.rack_rotation = Rotation.from_euler('z', rack_yaw_deg, degrees=True).as_matrix()
        self.rack_parts = []
        super().__init__(block_xy=block_xy, **kwargs)

    def _rigid(self, path, matrix, mass):
        # Build static colliders before the parent's first physics reset.
        if path.endswith('/Tweezers/Handle'):
            self._build_rack()
        return super()._rigid(path, matrix, mass)

    def _build_rack(self):
        x, y, z = self.rack_root_xyz
        top = self.table_height
        material = UsdShade.Material.Define(self.stage, '/World/Materials/Rack')
        shader = UsdShade.Shader.Define(self.stage, '/World/Materials/Rack/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.12, .38, .46))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.38)
        shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(.35)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        parts = [
            ('Base', (.12, .14, .012), (x, y+.045, top+.006)),
            ('RearPost', (.018, .018, z-top-.012), (x, y+.09, (z+top+.012)/2)),
            # The 4 mm-wide tongue fits between the two blades and supports
            # the underside of the joined tail. No tool attachment is used.
            ('SupportTongue', (.004, .105, .006), (x, y+.0425, z-.003)),
            ('GuideLeft', (.004, .024, .014), (x-.008, y, z+.007)),
            ('GuideRight', (.004, .024, .014), (x+.008, y, z+.007)),
            ('GuideBack', (.016, .004, .014), (x, y+.010, z+.007)),
            ('GuideFront', (.016, .004, .014), (x, y-.010, z+.007)),
        ]
        for name, size, position in parts:
            if name.startswith('Guide') and not self.rack_guides:
                continue
            position = np.asarray(self.rack_root_xyz) + self.rack_rotation @ (np.asarray(position)-self.rack_root_xyz)
            cube = self._box(self.env_paths[0]+'/Rack/'+name, size, position, self._table_mat, self.rack_rotation)
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
            self.rack_parts.append(dict(name=name, size_m=size, center_xyz_m=position.tolist(), rotation_matrix=self.rack_rotation.tolist()))
