"""Record the full physical demo through the verified real camera mounting poses."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def pose_matrix(position, wxyz):
    T = np.eye(4)
    T[:3, 3] = position
    T[:3, :3] = Rotation.from_quat(np.asarray(wxyz)[[1, 2, 3, 0]]).as_matrix()
    return T


class CalibratedDemoRenderer:
    def __init__(self, scene, rig_path, world_base_path, out, control_dt):
        import carb
        import imageio.v2 as imageio
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade
        from isaac_camera_rig import CalibratedCameraRig

        self.scene, self.out = scene, Path(out)
        self.frame_index = 0
        self.last_images = {}
        self.imageio = imageio
        base = json.loads(Path(world_base_path).read_text())
        if scene.num_envs != 1 or base['environment_index'] != 0:
            raise ValueError('Verified demo cameras currently support env_0 only')
        T_world_base = np.asarray(base['T_world_base'], dtype=float)
        i = scene.robot.body_names.index('base_link')
        data = scene.robot.data
        actual_base = pose_matrix(data.body_link_pos_w[0, i].cpu().numpy(),
                                  data.body_link_quat_w[0, i].cpu().numpy()) @ np.asarray(base['T_base_link_base'])
        if not np.allclose(actual_base, T_world_base, atol=2e-6):
            raise ValueError('Scene base differs from verified T_world_base; recompute it before recording')

        # Use the same visible lights/materials as the existing demo renderer.
        # No collision, articulation, pose or physics-material values are changed.
        UsdLux.DomeLight.Define(scene.stage, '/World/CalibratedDemoDome').CreateIntensityAttr(850.)
        key = UsdLux.DistantLight.Define(scene.stage, '/World/CalibratedDemoKey')
        key.CreateIntensityAttr(1200.)
        UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-25, 30, 0))
        for name, color, metal in [('RLSteel', (.5, .55, .62), .8),
                                    ('RLPlastic', (.95, .22, .03), 0),
                                    ('RLTable', (.16, .21, .27), .1)]:
            material = UsdShade.Material.Get(scene.stage, '/World/Materials/' + name)
            shader = UsdShade.Shader.Define(scene.stage, str(material.GetPath()) + '/Surface')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metal)
            shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.35)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        for suffix, name in [('/Table', 'RLTable'), ('/Block/Shape', 'RLPlastic'),
                             ('/Tweezers/Handle/Shape', 'RLSteel'), ('/Tweezers/Left/Blade', 'RLSteel'),
                             ('/Tweezers/Right/Blade', 'RLSteel'), ('/Tweezers/Left/Tip', 'RLSteel'),
                             ('/Tweezers/Right/Tip', 'RLSteel')]:
            prim = scene.stage.GetPrimAtPath(scene.env_paths[0] + suffix)
            if prim:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    UsdShade.Material.Get(scene.stage, '/World/Materials/' + name))
        carb.settings.get_settings().set('/rtx/post/aa/op', 0)
        self.rig = CalibratedCameraRig(rig_path, scene, T_world_base, image_mode='rectified')
        wrist = self.rig.rig['cameras']['wrist']
        self.T_link_tcp = self.rig.T_link_camera @ np.linalg.inv(np.asarray(wrist['T_TCP_Cw']))
        self.rig.update(0)
        self.fps = 1. / (2 * control_dt)
        self.writers = {name: imageio.get_writer(self.out / f'{name}.mp4', fps=self.fps,
                                               codec='libx264', quality=8, pixelformat='yuv420p')
                        for name in self.rig.cameras}
        self.records = (self.out / 'camera_frames.jsonl').open('w')
        self.time_origin = float(scene.sim.current_time)
        for _ in range(12):
            scene.sim.render()
        carb.settings.get_settings().set('/rtx/post/aa/op', 0)
        manifest = dict(image_mode='rectified', resolution=[640, 480], video_fps=self.fps,
                        T_world_base=T_world_base.tolist(),
                        rig=json.loads(Path(rig_path).read_text()),
                        source_files={str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                      for p in [rig_path, world_base_path]},
                        cameras={name: dict(video=f'{name}.mp4', first_frame=f'{name}_first.png',
                                            last_frame=f'{name}_last.png') for name in self.rig.cameras},
                        joint_names=scene.robot.joint_names,
                        renderer='Verified rig.rgb() native pinhole plus calibrated K resampling; AA off',
                        scope='Full original demo scene; calibrated camera mounting poses are not recentered on the task',
                        timestamp_note='Videos sample every other 60Hz control step; camera_frames.jsonl records exact simulation times, including the final frame')
        (self.out / 'camera_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print('CALIBRATED_CAMERAS_READY', str(self.out), flush=True)

    def frame(self):
        self.rig.update(0)
        self.scene.sim.render()
        images = self.rig.rgb()
        data = self.scene.robot.data
        i = self.rig.link_index
        T_world_link = pose_matrix(data.body_link_pos_w[0, i].cpu().numpy(),
                                   data.body_link_quat_w[0, i].cpu().numpy())
        for name, rgb in images.items():
            self.writers[name].append_data(rgb)
            if self.frame_index == 0:
                self.imageio.imwrite(self.out / f'{name}_first.png', rgb)
        row = dict(frame_index=self.frame_index, simulation_time_s=float(self.scene.sim.current_time),
                   elapsed_simulation_time_s=float(self.scene.sim.current_time) - self.time_origin,
                   joint_positions_rad=data.joint_pos[0].cpu().tolist(),
                   T_world_TCP=(T_world_link @ self.T_link_tcp).tolist(),
                   T_world_Cw=(T_world_link @ self.rig.T_link_camera).tolist(),
                   T_world_Cg=(self.rig.T_world_base @ np.asarray(self.rig.rig['cameras']['global']['T_B_Cg'])).tolist())
        self.records.write(json.dumps(row) + '\n')
        self.last_images = images
        self.frame_index += 1

    def close(self):
        for writer in self.writers.values():
            writer.close()
        self.records.close()
        for name, rgb in self.last_images.items():
            self.imageio.imwrite(self.out / f'{name}_last.png', rgb)
        (self.out / 'camera_recording.json').write_text(json.dumps(
            dict(frames=self.frame_index, videos=['wrist.mp4', 'global.mp4']), indent=2) + '\n')
