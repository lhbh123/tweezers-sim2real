# UR5e + XHand v15：历史基线与复现

当前 W01 工作入口见 [主 README](README.md)，本页仅描述保留的 XHand 历史基线。历史清理记录仍保留其当时日期和范围。


以下保留原 XHand 项目的实现、参数和验收数值，不能作为 W01 的结果。

**历史可复现版本：XHand v15，已完成当时的仿真验收与视频交付。** 文档整理日期：2026-10-19；实验完成日期：2026-10-17。

本项目从零实现任务代码与 PPO，使用 Isaac Lab 2.2.1 / Isaac Sim 5.0 的 GPU PhysX。机器人从**已经持有直尾镊子**的固定 HOME 姿态出发，接近桌面方块、夹起、搬运至指定目标、放下、松开并撤离。相同策略接受不同起点和目标，不需要为每个位置重新训练。

| 快速入口 | 用途 |
|---|---|
| [scripts/run_pick_place.py](scripts/run_pick_place.py) | 当前版本的训练、评估和原生录制入口 |
| [src/pick_place_task.py](src/pick_place_task.py) | 抓放任务、阶段控制、随机化、观测、奖励与成功判定 |
| [最终技术报告](reports/pick_place_revision_final.json) | 已完成实验、独立验收、物理审计和交付证据 |
| [当前进度记录](reports/pick_place_development_state.json) | `completed_and_delivered`，此版本无待完成验收项 |
| [对话与实验演进 HTML](outputs/tweezers_对话与实验演进记录_20261019.html) | 18 条需求、8 个阶段、10 段视频与 10 张图片；媒体内嵌，可离线查看 |

## 1. 最新功能与验证范围

- 起点与目标均在基座坐标系的 **X=[0.48, 0.76] m、Y=[−0.22, 0.08] m** 内连续采样；目标独立提出后重采样，直到起终点距离至少 12 cm。工作区为 **28 × 30 cm**，四角最大跨度约 41 cm。
- 使用 180 mm 直尾镊子，镊臂宽 14 mm、厚 1 mm；拇指和食指挤压两片镊臂相对的宽面。中指位于下方作为备用承托，中指和虎口不需要始终接触工具。
- 镊子倾斜向下，掌面不翻向上，允许虎口倾斜。v15 验收工具下倾范围为 20–70°；旧 v14 的 20–65° 与其不同。
- **几何分阶段控制 + 目标条件残差 PPO**：阶段控制器提供名义运动，PPO 根据 117 维仿真状态输出 6 维有界修正（腕部 XYZ、闭合协同量、食指、中指）。阶段顺序是稳持、接近、下降、夹紧、提起、搬运、落桌、松开、撤离、目标处稳定。
- 480 Hz 物理、60 Hz 控制。机器人关节驱动通过接触压缩被动弹簧镊臂；工具和方块的状态只在重置时写入，任务执行中没有固定附着、位姿搬运或外加抓取力。
- 镊尖离开已放下方块 20 mm 后，逐步将空镊握持闭合量恢复至 0.8，减少释放后的工具滑落。**复现选定策略必须传 `--retreat-closure .8`**；入口默认值为 0.7。

人类视频提供二维手部关键点、握持布局和朝向先验，再通过机器人运动学、接触和用户反馈调整。单目遮挡下的深度不可靠，没有将其当作精确人类三维动捕。v15 的 PPO 从随机网络训练，无 BC 初始化；历史 v14 的局部夹取采用过物理演示 BC + PPO，二者不要混称。

### 已完成的验收

| 验收 | 结果 | 原始记录 |
|---|---|---|
| 冻结模型后的独立随机测试 | **64/64**；平均落点误差 0.849 mm，中位数 0.515 mm，95 分位 2.818 mm，最大 3.384 mm | [evaluation.json](runs/pick_place15_ppo101_holdout20261017/evaluation.json)、[validation.json](runs/pick_place15_ppo101_holdout20261017/validation.json) |
| 四组边界对角搬运 | **4/4**；每组约 41 cm，最大误差 3.893 mm | [边界评估](runs/pick_place15_ppo101_four_corners/evaluation.json) |
| 近景录像逐物理子步几何检查 | 11,480 个子步通过关节限位、规定朝向、非相邻手部源表面相交及桌面间隙检查 | [子步审计](runs/pick_place15_ppo101_video_corner0/substep_geometry_audit.json) |

成功要求同时满足：双颚夹住并抬升至少 45 mm、保持 0.25 s；落点水平误差小于 10 mm、落桌高度误差小于 1.5 mm、线速度小于 10 mm/s、角速度小于 0.5 rad/s；双颚不再接触方块，镊尖撤离至少 35 mm；工具仍被正确保持，宽面接触及朝向合格，最终状态持续 0.75 s。

三组开发训练各 25 轮、153,600 个环境转移、400 次小批量更新，见 [训练证据](reports/pick_place15_training_evidence.json)。开发种子 51 上名义控制器 55/64、三组 PPO 候选各 63/64；候选 101 同时采用握持恢复，故这些数字**不是单独隔离 PPO 贡献的消融实验**。模型在独立验收前冻结，见 [冻结记录](reports/pick_place15_frozen_candidate.json)。

### 能力边界

这是有限测试集上的仿真结果，不代表整个平面或任意扰动下必然成功。当前使用仿真状态、预持镊初态、固定尺寸方块和已验证桌面区域；没有实现从桌面拿起镊子、视觉定位、未知物体适应或实机迁移。腕部以平移为主，没有学习任意腕旋转。机器人使用理想重力补偿，材料、摩擦和执行器尚未做实物标定。

中指备用承托由 v14 的配对松动试验验证，属于预设扰动下的被动承托，不是学习得到的主动防滑恢复。几何审计检查源网格与保存的离散状态，不等于对所有自碰撞、求解器内部连续时间状态或真实材料的证明。工具允许在放置时触桌，穿入检查容差为 0.1 mm。

## 2. 关键代码与调用关系

```text
scripts/run_pick_place.py
 ├─ src/rl_scene.py                 场景、关节驱动、被动镊子、接触采样
 ├─ src/pick_place_task.py          随机起终点、阶段控制、残差动作、抓放验收
 │   └─ src/rl_task.py              持镊、闭合协同、观测、姿态与接触指标
 │       ├─ src/reset_reference.py  稳定持镊快照初始化
 │       ├─ src/arm_command.py      腕部平移命令转关节目标
 │       └─ src/arm_clearance.py    初始化 IK 分支与桌面间隙
 ├─ src/ppo.py                     PPO 采样、GAE、归一化、更新、模型保存
 └─ src/probe_render.py            原生相机录制和镊尖跟随视角
```

| 文件 | 主要功能／修改位置 |
|---|---|
| [src/pick_place_task.py](src/pick_place_task.py) | `PickPlaceConfig` 定义工作区和控制参数；`reset` 采样起终点，`_nominal` 生成阶段动作，`step` 叠加策略残差与验收 |
| [src/rl_scene.py](src/rl_scene.py) | `BatchedTweezersScene` 创建 UR5e、XHand、桌子、方块、被动弹簧镊子，推进 PhysX 和读取状态 |
| [src/rl_task.py](src/rl_task.py) | `TweezerTask` 提供底层重置、关节控制、工具保持、宽面夹持和朝向指标 |
| [src/ppo.py](src/ppo.py) | `ActorCritic`、`RunningMoments`、GAE 与 PPO；训练、恢复训练和检查点序列化 |
| [src/grasp_contacts.py](src/grasp_contacts.py) | 接触角色、宽面内部区域、朝内法向受力与中指／虎口支撑统计；类名保留历史 `TripodContactReader` |
| [src/kinematics.py](src/kinematics.py) | 读取原始 URDF、正逆运动学与 STL 几何；机器人外部资产路径在此定义 |
| [src/arm_command.py](src/arm_command.py) | 命令空间 FK/Jacobian、增量 IK、关节速度及限位约束 |
| [src/arm_clearance.py](src/arm_clearance.py) | 基于源碰撞网格选择满足桌面间隙的 IK 解 |
| [src/reset_reference.py](src/reset_reference.py) | 将稳定快照变换到所需工作点，保留关节预紧及工具相对姿态 |
| [src/probe_render.py](src/probe_render.py) | 相机配置、原生视频与实际镊尖跟随；不修改物体运动 |
| [src/human_retarget.py](src/human_retarget.py) | 历史人类二维姿态先验到 XHand 关节参考及接触拟合；当前评估不逐帧运行它 |
| [src/run_tweezers.py](src/run_tweezers.py) | 最早的脚本控制基线，保留作历史对照，不是 v15 入口 |

验证与交付工具：

- [analysis/validate_pick_place.py](analysis/validate_pick_place.py)：从保存轨迹独立重构成功判定，并检查几何、USD 和源码来源。
- [analysis/convert_pick_place_substeps.py](analysis/convert_pick_place_substeps.py) + [analysis/audit_pick_place_substeps.py](analysis/audit_pick_place_substeps.py)：转换原始子步记录并分段检查全部物理样本；后者依赖 [audit_trained_tripod_geometry.py](analysis/audit_trained_tripod_geometry.py) 和 [validate_rl_hand_geometry.py](analysis/validate_rl_hand_geometry.py)，历史文件名不意味着仍要求持续三指支撑。
- [scripts/compose_pick_place_video.py](scripts/compose_pick_place_video.py)、[compose_pick_place_four.py](scripts/compose_pick_place_four.py)：将同一次真实运行的多视角录像合成近景或四案例视频。
- [analysis/finalize_pick_place_report.py](analysis/finalize_pick_place_report.py)：汇总模型、开发比较、独立验收与视频交付证据。
- [scripts/build_conversation_history.py](scripts/build_conversation_history.py)、[prepare_history_media.py](scripts/prepare_history_media.py)、[verify_history_html.py](scripts/verify_history_html.py)：构建、压缩嵌入媒体、验证离线历史报告。

## 3. 复现必需文件

| 用途 | 文件 |
|---|---|
| 当前手部姿态与镊子参数 | [configs/human_grasp_middle_backup_v14.json](configs/human_grasp_middle_backup_v14.json) |
| 稳定预持镊快照 | [runs/middle_backup13_level_search/m9_c0_snapshot.json](runs/middle_backup13_level_search/m9_c0_snapshot.json) |
| 选定模型 | [runs/pick_place15_ppo101_retreat80/iteration_00025.pt](runs/pick_place15_ppo101_retreat80/iteration_00025.pt) |
| 单案例／四角位置 | [pick_place_preview_case.json](configs/pick_place_preview_case.json)、[pick_place_boundary_cases.json](configs/pick_place_boundary_cases.json) |
| 单案例／四角相机 | [pick_place_cameras.json](analysis/pick_place_cameras.json)、[pick_place_four_cameras.json](analysis/pick_place_four_cameras.json) |

选定模型 SHA-256：`91523ab6e677e2d8ce85e655f0f29c6aad88d288ba2a710c703f749c48431fc6`。

**`runs/` 中含运行必需的快照和模型，不能把整个目录当缓存删除。** v15 继续使用 v14 的手部握姿，但任务、策略及验收范围已扩展。

外部机器人资产：`/cpfs/user/lvzhaoyang/work/ai_rl/assets/ur5e_xhand_right/robot.urdf` 及同目录网格；未复制进本项目，也不在清理范围。当前 RL 场景由 `rl_scene.py` 创建碰撞体；[assets/create_props_blender.py](assets/create_props_blender.py) 与 `assets/generated/` 保留早期道具资产生成链路。

## 4. 运行方式

从本项目根目录执行。仿真使用 `/cpfs/user/lvzhaoyang/apps/bin/isaac-python` 启动，以加载本机 Isaac Lab / Isaac Sim 环境；媒体与 HTML 工具使用 `.venv-human/bin/python`。此虚拟环境只用于人手估计／媒体，不替代仿真 Python。

使用物理 GPU 编号 `--device cuda:N`，不要另外设置 `CUDA_VISIBLE_DEVICES`。每次使用新的 `--out` 目录，入口会拒绝覆盖已有运行证据。同配置、同种子也不保证跨 GPU 或求解器版本逐位一致。

### 独立随机评估

```bash
/cpfs/user/lvzhaoyang/apps/bin/isaac-python scripts/run_pick_place.py \
  --out runs/my_v15_eval --num-envs 64 --seed 20261017 \
  --retreat-closure .8 --device cuda:0 \
  --evaluate runs/pick_place15_ppo101_retreat80/iteration_00025.pt

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 analysis/validate_pick_place.py \
  runs/my_v15_eval --geometry
```

### 指定起点和目标

创建 JSON，例如：

```json
{"start": [[0.55, -0.18]], "goal": [[0.72, 0.05]]}
```

评估命令增加 `--cases <JSON路径> --num-envs 1`。坐标为上述基座桌面坐标，单位米；批量输入行数必须与环境数一致。自定义输入应满足已验证工作区及至少 12 cm 的距离要求；显式案例不会自动替用户限制到可靠范围。

### 录制与合成完整回合

```bash
/cpfs/user/lvzhaoyang/apps/bin/isaac-python scripts/run_pick_place.py \
  --out runs/my_v15_video --num-envs 1 --seed 0 \
  --retreat-closure .8 --device cuda:0 \
  --evaluate runs/pick_place15_ppo101_retreat80/iteration_00025.pt \
  --cases configs/pick_place_preview_case.json \
  --camera-config analysis/pick_place_cameras.json --record-substeps

python3 analysis/convert_pick_place_substeps.py runs/my_v15_video
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 analysis/audit_pick_place_substeps.py \
  runs/my_v15_video --jobs 4
.venv-human/bin/python scripts/compose_pick_place_video.py runs/my_v15_video \
  --output outputs/my_v15_video.mp4
```

四角录制改用 `--num-envs 4 --cases configs/pick_place_boundary_cases.json --camera-config analysis/pick_place_four_cameras.json`，省略 `--record-substeps`；使用 `scripts/compose_pick_place_four.py <运行目录> --output <输出.mp4>` 合成四视图。

### 从随机网络重新训练

```bash
/cpfs/user/lvzhaoyang/apps/bin/isaac-python scripts/run_pick_place.py \
  --out runs/my_v15_train --num-envs 64 --seed 101 --iterations 25 \
  --learning-rate 1e-5 --std .04 --retreat-closure .8 --device cuda:0
```

`--evaluate zero` 运行名义几何控制器对照。`--resume <checkpoint>` 恢复训练状态；初始化参数与实际源码哈希保存到运行目录。新训练结果需重新评估，不自动继承这里记录的 64/64。

## 5. 视频、历史与目录

最新交付视频：

- [随机抓放近景 v15](outputs/ur5e_xhand_random_pick_place_closeup_v15.mp4)：完整回合全景、镊尖跟随近景与 XY 轨迹，原速。
- [四组边界搬运 v15](outputs/ur5e_xhand_random_pick_place_four_corners_v15.mp4)：四角预设边界案例，原速；不是随机样本录像。
- [中指备用承托对照 v14](outputs/ur5e_xhand_middle_backup_catch_v14.mp4)：相同松动控制下，中指在下方与收回的物理对照，0.5 倍速。

历史总结见 [离线 HTML](outputs/tweezers_对话与实验演进记录_20261019.html)。[整理前 README 快照](reports/README_history_before_cleanup_20261019.md) 留存历史说明，其部分探索路径已按本次清理移除，不能作为全部旧试验仍可运行的承诺。各代任务初态、成功条件不同，历史成功率不能直接当作同一基准的提升曲线。

| 目录 | 整理后的定位 |
|---|---|
| `src/` | 核心实现；本次未改变算法代码 |
| `scripts/` | 当前及关键历史版本入口、评估、视频和报告工具 |
| `configs/` | 当前配置及关键历史试验所需配置 |
| `analysis/` | 验证器、回归检查、接触／几何审计及少量仍有依赖的历史工具 |
| `runs/` | 当前训练、独立评估、原始媒体；关键历史模型与证据；清理时保护的少量旧试验目录 |
| `reports/` | 最终结论、历史证据、进度与交付记录、清理清单 |
| `outputs/` | 已整理交付的图片、视频与自包含 HTML |
| `references/`、`research/` | 人类视频、关键点、来源记录及研究材料 |
| `assets/` | 道具生成代码、已生成资产与字体 |
| `.venv-human/` | 保留的人手估计及媒体工具环境 |

## 6. 2026-10-19 清理记录

本次删除 **3,762 个文件，约 84.65 GB（78.84 GiB）**，包括 55 个一次性分析脚本和 31 个候选配置；`runs/` 子目录由 285 个减少到 66 个。保留 12 个核心源码文件，项目剩余约 10.70 GB（不含 `.venv-human/`）。

按用户要求删除未采用且不被保留入口／关键证据依赖的探索运行、一次性分析代码、候选配置、重复模型别名、开发场景导出和 Python 缓存。**实际删除文件与原因逐项记录在 [cleanup_manifest_20261019.json](reports/cleanup_manifest_20261019.json)**；该清单是审计记录，不是可恢复备份。

保留当前模型、预持镊快照、三个最终验收运行的完整原始数据与场景、历史报告输入和全部已交付媒体。部分开发比较运行只保留源码、启动参数、轨迹、结果和原验证记录，大型重复 USD 已删除；要重新审计该开发场景需用保存参数重新生成。历史探索记录中的旧路径可能指向已清理文件，应结合清理清单理解。

清理时发现 5 个旧仿真进程，因此先保留其相关脚本、输入和运行目录。用户随后要求确认能否停止：核查发现 4 个已保存最终结果但未退出，另 1 个无最终结果，全部日志超过 72 小时未更新且属于已替代的探索试验。2026-10-19 已对这 5 个准确匹配身份的进程发送 SIGTERM；全部退出，GPU 占用条目已清除，未使用 SIGKILL。已有结果、当前模型与最终报告哈希不变，详见 [进程停止记录](reports/legacy_process_shutdown_20261019.json)。本次停止未进一步删除文件。

此次整理不重训策略、不改变之前实验结果；整理后的依赖、模型和结果完整性检查见 [cleanup_validation_20261019.json](reports/cleanup_validation_20261019.json)。
