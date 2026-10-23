# 在仿真服务器确认 T_link_tcp

日期：2026-10-23。适用：UR5e + XHand、D405 腕部相机、D435 全局相机，Isaac Sim/Isaac Lab。

本文可随 `xhand_calibration_handoff_20261023.zip` 一起交给仿真服务器操作者。以下检查只在仿真中进行，不向真实机器人发送运动命令。

## 1. 目标与当前状态

真实相机内参、手眼、全局外参已求解，两路独立验证均通过。最终文件是：

- `results/wrist_intrinsics_v3.json`
- `results/global_intrinsics_v3.json`
- `results/handeye.json`
- `results/global_extrinsics.json`
- `results/validation_wrist.json`、`results/validation_global.json`

`results/camera_rig.json` 已汇总标定参数，但 `ready_for_isaac_wrist=false`，因为尚未确认目标仿真实际使用的连杆与真实 TCP 的关系。

本次真实 TCP 偏置为 `[0, 0, 0, 0, 0, 0]`。本地参考 URDF 推导出的 `T_wrist_3_link_TCP` 为单位阵；这是模型候选，不是目标 USD 已经核实的结果。不能只修改 ready 标志跳过确认。

最终需交付：

1. 已核对的 `configs/link_to_tcp.local.json`。
2. `results/camera_rig_isaac.json`。
3. 仿真使用的 `T_world_base` 及来源。
4. 实际资产路径/版本、运行时相对变换、多姿态对比误差和渲染检查记录。

## 2. 坐标约定

使用列向量、4×4 齐次矩阵、平移单位米：

```text
p_A = T_A_B @ p_B
```

| 符号 | 含义 |
|---|---|
| B | 真实 UR 控制器报告 TCP 位姿时使用的基座坐标系 |
| TCP | 标定时激活的工具坐标系 |
| link | 仿真适配器读取的刚体连杆坐标系，如 wrist_3_link |
| tool0 | 参考 UR 模型的零偏置工具坐标系 |
| Cw | D405 RGB 光学坐标系 |
| Cg | D435 RGB 光学坐标系 |
| Q | 本次固定 ChArUco 板坐标系 |

需要确认的矩阵：

```text
T_link_tcp = inverse(T_world_link) @ T_world_tcp
T_link_Cw = T_link_tcp @ T_TCP_Cw
T_world_Cw = T_world_link @ T_link_Cw
T_world_Cg = T_world_base @ T_B_Cg
```

UR 的 `[rx, ry, rz]` 是旋转向量（弧度）；URDF 的 `rpy` 是固定轴欧拉角，矩阵为 `Rz(yaw) @ Ry(pitch) @ Rx(roll)`。两者不可混用。

## 3. 检查实际加载的机器人资产

先在目标项目中确定真正加载的 USD/URDF 路径、机器人实例和环境编号。不要只根据目录名称判断。记录是否存在根节点缩放、固定关节合并、导入坐标调整。

包内 `model_reference/robot.urdf`、`ur_macro.xacro`、`robot.import.json` 是来源参考，不要求替换服务器现有机器人模型。import.json 中路径来自原导入环境，不应原样作为服务器资产路径使用。

本地参考 URDF 固定关节：

```text
wrist_3_link -> flange:
  xyz = [0, 0, 0]
  rpy = [0, -pi/2, -pi/2]

flange -> tool0:
  xyz = [0, 0, 0]
  rpy = [pi/2, 0, pi/2]
```

参考模型源码将 flange 定义为末端安装坐标系，将 tool0 定义为 all-zeros tool frame。因此：

```text
T_wrist3_tool0 = T_wrist3_flange @ T_flange_tool0 = I
```

本次真实 TCP 为零偏置，在这个模型约定下 `T_tool0_tcp=I`，故候选 `T_wrist3_tcp=I`。不要把 URDF 名为 flange 的坐标系直接当作控制器零工具坐标系。

若以后 TCP 非零：

```python
import numpy as np
from scipy.spatial.transform import Rotation

offset = np.asarray(actual_tcp_offset_m_rotvec, dtype=float)
T_tool0_tcp = np.eye(4)
T_tool0_tcp[:3, :3] = Rotation.from_rotvec(offset[3:]).as_matrix()
T_tool0_tcp[:3, 3] = offset[:3]
T_link_tcp = T_link_tool0 @ T_tool0_tcp
```

改变真实 TCP 后，旧标定的参考系也随之变化，不可只改导出配置后继续假定旧手眼结果适用。

## 4. 在实际 USD 中检查固定相对变换

在已启动的 Isaac Sim 中加载目标场景。以下代码放到 Script Editor 或已启动 SimulationApp 的脚本中运行；不是普通 Python 终端可直接执行的完整启动脚本。

先查找 wrist_3_link、tool0、base 的实际 prim 路径。本地导入参考路径为：

```text
/ur5e_xhand_right/root_joint/wrist_3_link
/ur5e_xhand_right/root_joint/wrist_3_link/__frames/tool0
/ur5e_xhand_right/root_joint/base_link/__frames/base
```

服务器若使用 `/World/envs/env_0/...`，必须替换成对应实例路径。tool0 可能是合并后的普通 Xform，不在 articulation 的刚体列表中，这是正常的。

```python
import numpy as np
import omni.usd
from pxr import Usd, UsdGeom

stage = omni.usd.get_context().get_stage()
assert np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0), '先统一场景为米'

# 替换为实际路径。
LINK_PATH = '/ur5e_xhand_right/root_joint/wrist_3_link'
TOOL_PATH = '/ur5e_xhand_right/root_joint/wrist_3_link/__frames/tool0'

def world_transform_column(path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise ValueError(f'Prim 不存在: {path}')
    # USD Gf.Matrix 使用行向量约定；转置为本文列向量约定。
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    T = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float).T
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5):
        raise ValueError('变换含缩放/剪切，不能直接作为刚体姿态')
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-5):
        raise ValueError('旋转行列式不为 +1')
    return T

T_world_link = world_transform_column(LINK_PATH)
T_world_tool0 = world_transform_column(TOOL_PATH)
T_link_tool0 = np.linalg.inv(T_world_link) @ T_world_tool0
print('T_link_tool0 =\n', T_link_tool0)
```

对参考模型应接近单位阵。若不是，先检查实际资产和坐标轴，不要强行覆盖成单位阵。

运行物理仿真时，USD Xform 可能未同步 PhysX 的实时姿态。上述方法适合核对固定局部链；动态多姿态检查必须使用同步的运行时姿态，不能混用更新后的 articulation 状态和陈旧的 USD 世界矩阵。必要时用固定局部链构造 `T_world_tool0 = T_world_link_runtime @ T_link_tool0`。

## 5. 核对适配器实际读取的连杆坐标

仓库适配器通过 `scene.robot.body_names` 查找 reference_link，并读取 `body_pos_w/body_quat_w`。确认目标框架这些字段代表连杆原点，而不是刚体质心；不同版本如提供 `body_link_pos_w/body_link_quat_w`，应按版本文档核实。读取坐标必须和确认 T_link_tcp 时使用的坐标一致。

现有适配器的读取方式如下（scene 已初始化，物理状态已更新）：

```python
from scipy.spatial.transform import Rotation

env_id = 0
idx = scene.robot.body_names.index('wrist_3_link')
data = scene.robot.data
p = data.body_pos_w[env_id, idx].detach().cpu().numpy()
q = data.body_quat_w[env_id, idx].detach().cpu().numpy()  # wxyz
T_world_link_runtime = np.eye(4)
T_world_link_runtime[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
T_world_link_runtime[:3, 3] = p
```

改变几个仿真关节姿态后，同步计算的 `inverse(T_world_link) @ T_world_tcp` 应保持不变。固定链检查的数值误差应接近计算精度；这与真实机器人运动学误差是两回事。

如果 tool0 被导入器删除，需要从实际导入来源重建固定关节链，并验证导入是否保持连杆坐标。不能因为找不到 tool0 就默认单位阵。

## 6. 对齐控制器基座，并用真实状态交叉验证

参考 URDF 中：

```text
base_link -> base:
  xyz = [0, 0, 0]
  rpy = [0, 0, pi]
```

这里的 base 对齐 UR 控制器基座。因此在参考模型中：

```python
T_base_link_base = np.diag([-1.0, -1.0, 1.0, 1.0])
T_world_base = T_world_base_link @ T_base_link_base
```

若实际 USD 保留了正确 base frame，可直接读取其同步世界变换。不要重复应用 180° 旋转。T_world_base 应包含选定克隆环境的位置偏移。

回传包有 `data/validation_wrist/samples.jsonl`，每行包含真实六关节角和 TCP 位姿：

```python
import json
from pathlib import Path
samples = [json.loads(line) for line in
           Path('data/validation_wrist/samples.jsonl').read_text().splitlines()
           if line.strip()]
s = samples[0]
q_real = np.asarray(s['robot']['joint_positions_rad'])
T_B_TCP_real = np.asarray(s['robot']['T_B_TCP'])
print(s['sample_id'], q_real, T_B_TCP_real)
```

UR 的六关节顺序通常对应：

```text
shoulder_pan_joint, shoulder_lift_joint, elbow_joint,
wrist_1_joint, wrist_2_joint, wrist_3_joint
```

在目标仿真中按实际关节名称映射，不能直接写入整个机器人前六维：XHand 关节也在 articulation 中。用项目现有接口设置仿真关节姿态，保持手部状态，更新物理/运动学和数据缓存后读取实际关节角，确认与目标值一致。这里不提供通用写入调用，避免不同框架接口及关节顺序差异。

随后计算：

```python
T_world_TCP_sim = T_world_link_runtime @ T_link_tcp
T_B_TCP_sim = np.linalg.inv(T_world_base) @ T_world_TCP_sim
translation_error_mm = np.linalg.norm(
    T_B_TCP_sim[:3, 3] - T_B_TCP_real[:3, 3]) * 1000
rotation_error_deg = np.degrees(Rotation.from_matrix(
    T_B_TCP_real[:3, :3].T @ T_B_TCP_sim[:3, :3]).magnitude())
print(translation_error_mm, rotation_error_deg)
```

建议至少比较 5 个不同验证姿态，保存 sample_id、实际仿真关节角、平移误差和旋转误差。不要为了对上一帧而临时修改基座/相机，随后又在另一帧更换变换。

这里没有自动通用的通过阈值：真实 UR 个体运动学校准与通用 URDF 可存在差别，应结合项目所需精度判断。原相机验证的 5 mm/2° 阈值不自动成为机器人模型核对标准。明显的 90°/180° 差异、厘米级以上的系统偏移，应先查坐标、关节和资产，再分析个体运动学误差。

## 7. 确认并导出

仅在实际模型和运行时坐标核对完成后，创建 `configs/link_to_tcp.local.json`。

如果已确认实际模型与参考模型一致、读取 wrist_3_link 原点、本次 TCP 为零，内容可为：

```json
{
  "reference_link": "wrist_3_link",
  "verified": true,
  "T_link_tcp": [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ],
  "note": "Actual simulation link/tool0 frames and zero configured TCP checked; see verification record."
}
```

如果实际模型关系不同，填写实际确认的矩阵。reference_link 必须是适配器可以从 body_names 中找到的连杆名称；非刚体的 tool0 frame 不能直接替代它而不修改适配器。

在安装了本标定项目的 Python 环境中，从回传包根目录执行：

```bash
python -m xhand_calib export \
  --wrist-intrinsics results/wrist_intrinsics_v3.json \
  --global-intrinsics results/global_intrinsics_v3.json \
  --handeye results/handeye.json \
  --global-result results/global_extrinsics.json \
  --link-to-tcp configs/link_to_tcp.local.json \
  --out results/camera_rig_isaac.json
```

输出路径必须不存在。需要安装导出工具时，可在独立 Python 3.10–3.12 环境中执行 `python -m pip install -e .`，无需 hardware 依赖；不要为导出修改已经跑通的 Isaac Sim 环境。也可把确认后的 link_to_tcp.local.json 传回采集机完成导出。

检查导出字段：`ready_for_isaac_wrist=true`、正确的 reference_link、有效的 T_link_Cw。ready 标志只表明提供了确认配置，不是自动完成了物理验证。

## 8. 最后的渲染对齐检查

使用 `examples/isaac_camera_rig.py`，并提供已确认的 T_world_base。建议 `image_mode='rectified'`，真实图片按相同 alpha=0 去畸变，640×480，全幅不裁剪。

使用验证样本机器人关节角，在仿真中将板坐标系放在：

```text
T_world_Q = T_world_base @ handeye['T_B_Q']
```

板尺寸为 7×10 格，单格 0.027366667 m，标记边长 0.019704 m。板资产的原点、坐标轴、图案正反面必须与 OpenCV ChArUco Q 坐标定义一致，不能把板中心直接当作 Q 原点。

比较真实与渲染的板角点：中心、四周、画面方向、腕部旋转后的随动。适配器已经负责 OpenCV 光学轴到 USD 相机轴的转换，不要重复转换。每次物理状态更新后调用 rig.update，再渲染取图。

## 9. 常见偏差

| 现象 | 优先检查 |
|---|---|
| 固定差 90°/180° | flange/tool0 混用、base/base_link 混用、四元数顺序 |
| 平移差随腕部旋转而改变 | TCP 偏置、连杆与质心混用、错误的安装平移 |
| 换姿态后相对固定变换也变化 | USD/PhysX 状态未同步、读错实例、矩阵乘法方向 |
| 所有位置差约 1000 倍 | 毫米/米或 USD 单位错误 |
| 机器人 TCP 对齐但相机画面反向 | 光学轴转换、相机约定或重复轴转换 |
| 图像中心较好但边缘偏差大 | 原始/去畸变模式、K/D、分辨率、裁剪和镜头模型 |
| 一个姿态对齐，其他姿态偏差大 | 关节映射、运动学模型或用单姿态补偿错误 |

## 10. 建议回传的确认记录

```text
实际 USD/URDF 路径和版本/hash：
环境编号：
reference_link 及实际 prim 路径：
tool0 prim 路径或固定链来源：
读取的姿态 API，连杆原点还是质心：
真实 TCP 偏置：
T_link_tcp：
T_world_base 及 base/base_link 转换来源：
多个姿态固定变换检查：
真实/仿真 TCP 对比误差：
渲染角点对齐结果：
仍未解决的偏差：
```
