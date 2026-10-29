# 已标定场景中的连续取镊与夹放测试

入口为 `scripts/test_tweezer_pipeline.sh`。在同一个物理场景中从固定起点取镊，保留真实抓取状态，切换到原冻结夹物 PPO 执行抓取、搬运与放置。切换前后逐项比较关节位置/速度与镊子、物块状态，任何非零变化都会报错；不会用稳定快照覆盖取镊结果。

## 与原 Demo 对齐的默认配置

| 内容 | 来源或数值 |
|---|---|
| 桌面、HOME | `simulation/configs/calibrated_demo_y_table.json` |
| 物块起点、放置目标 | `simulation/configs/calibrated_demo_y_case.json` |
| 物块起点 XY | `(0.22, 0.48)` m，中心初始 Z 为桌高加 6.1 mm |
| 放置目标 XY | `(-0.08, 0.76)` m |
| 桌面尺寸/中心/高度 | `0.80 × 1.10 × 0.04` m；中心 XY `(0.0, 0.35)`；顶面 Z `0.72` m |
| 腕部与全局相机 | `calibration/results/camera_rig_isaac.json` |
| 仿真基座与标定基座关系 | `calibration/results/world_base_isaac.json` |
| 图像生成 | 复用 `CalibratedDemoRenderer`，rectified RGB，640×480，30 FPS |
| 物理设置 | 原 `run_calibrated_demo.sh` / `run_pick_place.py` 的默认配置 |
| 支架与镊子初始位置 | 在取镊训练布局上沿世界 +Y 平移 0.10 m；根位置为 `(0.20, 0.62, 1.00)` m |

物理参数从回合开始就统一使用：弹簧刚度 5、阻尼 0.02；位置/速度求解迭代 64/8；接触偏移 0.0002 m；不简化机械臂碰撞网格；手部驱动刚度/阻尼 10/0.3、手指摩擦 1.4。交接时不会切换物理参数。

在原夹放场景上新增被动支架和取镊前半程，支架仍取自 `tweezer_rack_scene.json`。这里不会沿用该文件中的 `block_xy_m` 或展示相机 `views`；桌面布局与物块起终点分别以原 Demo 的 layout/cases 文件为准。目标由四条绿色可视标记表示，不添加碰撞体。相机保持原标定安装位置，不根据新增支架重新居中。

独立取镊训练使用不同的求解器、接触设置、物块起点与支架位置，因此完整 pipeline 是在调整后的场景条件下重新评估取镊模型。入口继续严格检查任务、源码签名、支架形状/旋转、HOME 与任务参数；允许显式声明的支架整体 Y 平移，以及用户要求恢复的物块起点，并记录 `pickup_evaluation_overrides.json`。逐个支架部件检查其中心只能发生指定的共同 Y 位移；其他几何差异仍拒绝加载。这种评估覆盖不表示模型已在新场景下训练或保证成功。

`--rack-offset-y` 默认为 `0.10`（米），同时移动整个支架和初始镊子；`--rack-offset-y 0` 恢复原先 Y=0.52 m 的布局。机械臂固定初始关节角、物块、目标、HOME 和标定相机不随之平移。独立取镊训练配置文件保留原值。新布局的物块与底座前缘间隙约为 109 mm，原布局约为 9 mm。

## 命令

```bash
cd /cpfs/user/xiaoqihuang/tweezers-sim2real
bash scripts/test_tweezer_pipeline.sh \
  --out "runs/tweezer_pipeline_calibrated_$(date +%Y%m%d_%H%M%S)" \
  --pickup-checkpoint runs/rack_pickup_fixed_pilot_01/latest.pt \
  --place-checkpoint ../models/policies/xhand_pick_place_ppo.pt \
  --episodes 1 --video --device cuda:0
```

输出目录相对 `simulation/`，必须不存在。默认起终点无需另传参数。可通过 `--cases`、`--scene-layout` 指定其他案例；只有显式传入 `--goal-xy` 才覆盖案例文件中的目标。

只查看初始布局时，用 `--preview-only` 替代 `--video`：仍加载模型并检查兼容性，输出标定双相机的 `*_first.png`、初始状态和来源检查，但不执行任务、不生成成功判定。预览输出中的单帧视频不是完整流程录像。

## 输出与判定

- `evaluation.json`：逐回合 `pickup_success`、`place_success`、总流程 `success`，各阶段失败/超时、目标误差。
- `scene_layout.json`：实际起终点、桌面与物理参数。
- `pickup_evaluation_overrides.json`：训练与完整测试场景的已知差异。
- `episode_000/wrist.mp4`、`global.mp4`：同一回合从取镊到结束的标定双视角视频。
- `episode_000/camera_manifest.json`、`camera_frames.jsonl`：相机标定内容、精确仿真时间与控制步索引。
- `episode_000/handoff.json`：交接时实际状态和连续性检查。
- `episode_000/steps.jsonl`：任务阶段、物块抬升高度、镊片—物块接触力、双侧夹持、握姿、镊尖目标误差等诊断量。
- `episode_000/trajectory.npz`：连续关节/工具/物块/动作轨迹；stage 0 为取镊，stage 1 为夹放。
- `source/`、`checkpoints/`、`launch.json`、`source_check.json`：源码和模型快照与运行边界检查。

之前的 `tweezer_pipeline_eval_20261028_205302` 使用展示相机、物块起点 `(0.0, 0.65)` 及取镊训练的物理设置，属于旧测试条件。其取镊通过，但夹放在 lift 阶段超时，尚不能据此评判恢复原场景后的成功率。

## 2026-10-28 原支架位置的完整验证

输出为 `simulation/runs/tweezer_pipeline_calibrated_20261028_complete/`，使用与旧失败测试相同的两个模型权重，固定案例完整成功 **1/1**。该次支架根位置为 `(0.20, 0.52, 1.00)` m，对应当前参数 `--rack-offset-y 0`；不应将此结果当作平移后的验证。

| 检查 | 实际结果 |
|---|---|
| 取镊 | 成功，18.633 秒 |
| 夹取、搬运、释放及目标处稳定 | 成功，23.533 秒 |
| 全流程仿真时间 | 42.167 秒，2530 个控制步 |
| 最终物块 XY 误差 | 0.726 mm |
| 最大物块抬升 | 84.281 mm，相对初始中心高度 0.7261 m |
| lift / transfer / lower 双侧夹持 | 各阶段所有记录步均为 true |
| 失败、阶段超时、机械臂—支架碰撞 | 均未触发 |
| 交接连续性 | 六项关节/物体状态最大变化均为 0；未恢复快照 |
| 场景及相机对照 | 桌面、HOME、起终点与原 Demo 一致；相机 rig 和 T_world_base 完全相同 |
| 双相机视频 | 各 1266 帧，640×480、30 FPS，逐帧解码完整 |
| 数值及来源检查 | 轨迹全部有限；运行期间跟踪源码与配置哈希未变化 |

完整记录见 [evaluation.json](../simulation/runs/tweezer_pipeline_calibrated_20261028_complete/evaluation.json)，离线核验见 [verification.json](../verification/tweezer_pipeline_calibrated_20261028/verification.json)，视频见 [全局](../simulation/runs/tweezer_pipeline_calibrated_20261028_complete/episode_000/global.mp4) / [腕部](../simulation/runs/tweezer_pipeline_calibrated_20261028_complete/episode_000/wrist.mp4)。这些本地运行产物不随 Git 分发。

本次同时恢复了场景布局和原 Demo 物理参数，尚未单独消融各项差异，因此不能将旧测试失败归因于某一项参数。单案例成功不代表随机场景总体成功率，也不代表真机精度。较早的 `tweezer_pipeline_calibrated_20261028/` 在录制期间中断，没有最终判定，已在 2026-10-29 清理；其小型元数据及删除清单见 [清理记录](../verification/runs_cleanup_20261029/README.md)。应查看带 `_complete` 后缀的完整结果。

## 2026-10-28 支架与镊子沿 +Y 平移 10 cm

用户要求增大支架与物块间距后，完整测试默认增加 `--rack-offset-y 0.10`。支架和镊子根位置为 `(0.20, 0.62, 1.00)` m，形状、朝向和高度保持原值。物块、目标、HOME、固定初始关节姿态和相机不变。

使用现有两个模型完成无录像的整回合验证，输出为 `simulation/runs/tweezer_pipeline_rack_y062_check_20261028/`：固定案例成功 1/1，取镊 21.567 秒、夹放 23.517 秒、总计 45.083 秒，最终 XY 误差 0.616 mm。没有触发阶段超时或机械臂与支架碰撞；交接六项状态差异均为零，运行来源哈希检查通过。该次结果没有完整视频。

初始场景预览另存于 `simulation/runs/tweezer_pipeline_rack_y062_preview_20261028/episode_000/`。原支架位置的成功视频仍保留在上述 `_complete` 目录中。
