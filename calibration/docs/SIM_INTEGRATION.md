> 本文描述已验证相机适配器的接口。当前已经接入 simulation/；运行命令见 [根 README](../../README.md)。以下原确认流程中的外层报告现存于 archives/calibration_confirmed_20261023/。

# 接入现有 tweezers 仿真

目标项目是 `tweezers_xhand_w01/tweezers`，现有相机位于 `src/probe_render.py`。本仓库不自动改写该项目，也不重新训练 PPO。

## 先完成真实标定

完成 README 的独立验证后，准备：

1. `results/camera_rig_isaac.json`，其中 `ready_for_isaac_wrist=true`。
2. 仿真机器人**同一个实际基座坐标系**在世界中的 `T_world_base`。
3. `reference_link`，通常是当前项目可读取的 `wrist_3_link`，但以模型为准。

真机配置 TCP 不自动等于 `wrist_3_link`。对于本次 UR 模型，控制器零工具坐标对应 `tool0`；若激活的 TCP 偏置表示为 `T_tool0_tcp`，则：

```text
T_link_tcp = T_link_tool0 @ T_tool0_tcp
```

URDF、控制器 base/base_link 的约定也可能不同。`T_world_base` 应对应真实控制器位姿使用的基座坐标系，不能只看名称相似就设为单位阵。克隆环境下，全局相机的位置也需要包含选定环境的平移。

## 使用适配器

将 `examples/isaac_camera_rig.py` 和 `examples/isaac_camera_bootstrap.py` 复制到目标项目的 `src/`，或将 examples 目录加入 Python 路径。它只依赖仿真环境已有的 NumPy、SciPy、Isaac Sim 相机 API，不需要在仿真环境安装 RealSense、RTDE 或 OpenCV。

Isaac Sim 5.0 的独立启动需要提前注册镜头 USD schema，否则在创建相机时会找不到 `OmniLensDistortionOpenCvPinholeAPI`。在入口脚本中，**先于导入 AppLauncher、omni、pxr**执行：

```python
from isaac_camera_bootstrap import register_lens_schemas
register_lens_schemas()
```

随后按项目原方式创建 AppLauncher 参数，并在启动前设置：

```python
args.headless = True
args.enable_cameras = True
args.kit_args += " --enable omni.usd.schema.omni_lens_distortion"
app = AppLauncher(args).app
```

适配器会启用 `isaacsim.sensors.camera`。本次经过验证的 `rectified` 模式使用原生针孔投影与 NumPy 双线性重采样。只有尚未验证的 `raw` 分支使用 OpenCV 镜头 schema，并显式设置 `imageSize`，避免继承 2048×1024 默认值。必须先启动 `SimulationApp`，再导入/构造适配器。`scene` 初始化并建立机器人状态后：

```python
from isaac_camera_rig import CalibratedCameraRig

# 由你的场景坐标配置计算，不要凭空填写单位阵。
T_world_base = measured_robot_base_transform_in_sim_world

rig = CalibratedCameraRig(
    rig_path="/absolute/path/to/camera_rig_isaac.json",
    scene=scene,
    T_world_base=T_world_base,
    image_mode="rectified",
)

# 仿真循环中：完成 physics/robot state update 后，再渲染。
rig.update(env_id=0)
scene.sim.render()
images = rig.rgb()  # {'global': RGB ndarray, 'wrist': RGB ndarray}
```

这是与现有 `scene` 对象衔接的代码片段，不是可以脱离 SimulationApp 单独运行的脚本。创建后可能需要按原项目方式 render 几帧预热，再取图。

不再对腕部相机执行原来的 `follow_tip`。适配器每帧读取指定连杆的世界位置/四元数：

```text
T_world_wrist_camera = T_world_link @ T_link_Cw
T_world_global_camera = T_world_base @ T_B_Cg
```

默认 `rectified` 与 README 的 `rectify --alpha 0` 相匹配：相同分辨率、新 K、零畸变、不裁剪。若选择 `raw`，适配器使用拟合的 OpenCV K/D。

## 关于 Isaac Sim 内参设置

本次验证的 `rectified` 路径采用以下步骤：

1. 先用 `f=min(fx,fy)` 的方形像素针孔相机渲染，以覆盖目标图像所需视野。
2. 用 aperture offset 设置主点，并处理渲染像素中心与 OpenCV 像素中心的半像素约定。
3. 按 `K_target @ inverse(K_native)` 进行 NumPy 双线性重采样，输出仍为 640×480、目标 K、零畸变、不裁剪。

这是根据内参计算的确定映射，不依赖某个机器人姿态或拟合单帧误差。当前渲染器会按方形像素处理原生投影，直接分别填写 fx/fy 对应的两个 aperture 不能保证输出符合非等焦距 K。因此正式取图应使用 `rig.rgb()`；直接调用底层 `camera.get_rgba()` 得到的是尚未重采样的中间图像。

适配器用较大的等比例虚拟焦距/感光面尺寸避免小数值焦距的原生渲染范围问题；这只影响参数尺度，视野由两者比值决定，不能把该虚拟焦距当作 RealSense 的实测物理焦距。

OpenCV 光学轴转换为 USD 的 +Y 向上、-Z 观察方向，四元数使用 Isaac 的 wxyz。`raw` 分支仍保留 `set_opencv_pinhole_properties` 路径，但不属于本次已验证功能。

本次版本使用实际项目资产、真实验证姿态和 640×480 标定板执行了渲染检查，数值与图片见本交付目录外层的 `CONFIRMATION_REPORT.md` 和 `verification/`。更换模型、分辨率或环境位置后仍应复查：

- 中心和四角是否一致；
- 非中心主点是否正确生效；
- 画面是否上下翻转或朝反方向；
- 转动腕部时视图是否随连杆正确旋转；
- 镜头近裁剪面是否切掉近距离物体。

若底层渲染器对 fx/fy 或镜头畸变属性的处理与你的预期不同，以标定板投影验证为准，不通过缩放物体来补偿相机错误。

## 同步与保存

每次物理更新后更新腕部位姿，再执行一次 render，随后取两路图像。两路真实相机和仿真数据的帧率可以分别记录，不应把 30 FPS 视频与控制频率自动视为一一对应。

`rig.rgb()` 返回 RGB；若用 OpenCV 保存：

```python
cv2.imwrite("global.png", cv2.cvtColor(images["global"], cv2.COLOR_RGB2BGR))
```

原来的 ProbeRenderer 主要负责视频诊断。正式数据采集建议另外保存两路 PNG、同一时刻的仿真状态和相机位姿；本仓库提供相机对齐适配器，没有自动修改原 demo 的数据集格式。

## 本次确认配置

使用 `results/camera_rig_isaac.json`，不要使用原始未确认的 `results/camera_rig.json`。
`reference_link=wrist_3_link`，读取 `body_link_pos_w/body_link_quat_w`（连杆原点，wxyz），`T_link_tcp=I`。
`results/world_base_isaac.json` 对应 env_0、机器人平移 [0,0,0.72] 米；更换环境原点、台面高度或机器人基座安装方向时需要重新计算。
实际变换及哈希、十姿态运行时误差在外层验证记录中。

投影复查脚本关闭 `/rtx/post/aa/op`（值 0）以避免小图案被超分辨率处理影响。实际数据采集若重新启用 DLSS/其他后处理，请重新核对清晰度。小尺寸 ChArUco 的渲染识别数量不足，本次渲染验收使用另外一遍图像中的六种独立颜色球心，不据此声称完成了渲染标签可读性验证。
