# W01 自然握姿与官方 TacMap：基线、改进与边界

## 2026-10-20 柔顺接触改进

新增独立入口 [run_sharpa_w01_compliant.sh](scripts/run_sharpa_w01_compliant.sh)。力与 TacMap 共用 matched_v2 表面；旁路保存官方原始 CPD 和量化过程；皮肤启用 280/20 柔顺接触，手与镊子 rest offset 均为 0。重新建立预持镊快照和有界闭合协同，并在起抬前要求真实双侧夹块力各达到 0.0604 N、连续 0.25 s。原摩擦、镊子弹簧、关节力矩和限位保留。

**这是实验版本，尚未达到旧任务的鲁棒性。** 固定案例完整记录通过：最大提升 **83.969 mm**、最终落点误差 **1.153 mm**、25.25 s。相同八组起终点回归为 **5/8，旧刚性基线为 8/8**；失败分别为夹紧不足、抬起后掉落、最终位置/宽面条件不足。旧默认入口保留。

**不能把食指变亮直接当成正确压入。** 新运行的官方多目标原图食指峰值为 241，但独立解析发现正反射线会跨两片镊臂，错误接受盒外点。例如搬运 13.3167 s，原食指 11232 个正深度点中，8337 个明确在物体外；逐刚体分别执行 CPD 判据后，正点为 2893，灰度峰值由 240 降至 153。原始数据保留，逐目标复算另存并标为场景适配扩展，不能称为官方原始输出或实物标定。

完整 **758 帧×5 指**的逐目标离线复算已完成：五指全程峰值为 **115/155/123/0/0**；每个刚体的正CPD起点均在该刚体源盒内，最小正值与最大值合并在本次轨迹逐像素相同，原始NPZ哈希不变。[完整复算与边界说明](sharpa_w01_tactile_validation/improvement_validation/per_target_replay/full_trajectory/report.json)单独保存。这不保证非凸或不连通目标的普适正确性，也不是重新运行了一次物理仿真。

所有运行、失败试验、独立审查和媒体保存在 [improvement_validation](sharpa_w01_tactile_validation/improvement_validation/)；见 [报告](sharpa_w01_tactile_validation/improvement_validation/IMPROVEMENT_VALIDATION_REPORT.txt)、[PDF](sharpa_w01_tactile_validation/improvement_validation/improvement_validation报告.pdf)、[固定案例](sharpa_w01_tactile_validation/improvement_validation/runs/compliant_v1_full_validation_v2/)、[八例回归](sharpa_w01_tactile_validation/improvement_validation/review/replay8_comparison.json)。完整任务有 12120 个物理子步、758 个五指图帧；重构任务判据、关节限位、源哈希及标准传感器验证通过。标准传感器验证只证明官方计算链路一致，不能排除上述多对象几何歧义。

```bash
scripts/run_sharpa_w01_compliant.sh --out runs/NEW_W01_COMPLIANT \
  --cases configs/sharpa_w01_hard_case.json --device cuda:0 \
  --camera-config configs/sharpa_w01_tacmap_cameras.json \
  --tactile --tacmap --tacmap-diagnostics --collision-geometry --contact-shapes

.venv-human/bin/python sharpa_w01_tactile_validation/improvement_validation/scripts/replay_per_target_tacmap.py \
  runs/NEW_W01_COMPLIANT --output runs/NEW_W01_COMPLIANT_per_target
```

输出目录必须不存在。第二步使用已保存的真实位姿离线复算，原始物理轨迹和官方图均不改写。柔顺方案仍从预持镊开始；全程严格宽面夹持、皮肤/壳体接触力归属、硬件标定和更广泛的任务稳定性尚未解决。

## 以下为此前刚性基线的冻结结果与分析

此前最终演示为 **`sharpa_w01_official_tacmap_matched_v2_video`**：UR5e 搭载 Sharpa Wave W01 右手，以拇指、食指夹持镊臂宽面，中指在下方辅助承托，无名指、小指自然弯曲。释放时中指暂时退让，离块后恢复。运行从已经持镊的物理快照开始，不包含从桌面拿起镊子。

这次完成了**实际调用官方 TacMap 算法的几何触觉演示**，同时记录独立的 PhysX 接触力。**尚未完整对齐官方压入示例的物理设置，也未验证实机力—形变关系。** 当前仍使用刚性接触、零 rest offset；不能把图像灰度解释为压力、实物皮肤形变或承托力。

[最终视频](outputs/sharpa_w01_official_tacmap_matched_v2.mp4) · [响应曲线](outputs/sharpa_w01_official_tacmap_response.png) · [传感器压入对照](outputs/sharpa_w01_official_tacmap_matched_controls.png) · [最终报告](reports/sharpa_w01_official_tacmap_final.json)

## 复现当前结果

环境为 **Isaac Lab 2.2.1 / Isaac Sim 5.0**。从项目根目录运行，输出目录必须不存在；`--tacmap` 和 `--tactile` 当前只支持单环境记录。

```bash
scripts/run_sharpa_w01.sh --out runs/sharpa_w01_tacmap_new \
  --pose configs/sharpa_w01_natural_v6d_tacmap_matched_v2.json \
  --cases configs/sharpa_w01_hard_case.json \
  --camera-config configs/sharpa_w01_tacmap_cameras.json \
  --tacmap --tactile --record-substeps --episode-seconds 40 --device cuda:0
```

[W01 入口](scripts/run_sharpa_w01.sh) 默认使用上述姿态和 [v6c 稳定快照](configs/sharpa_w01_natural_v6c_preheld.json)，默认执行阶段控制器 `--evaluate zero`。默认参数包括闭合系数 open/closed/retreat = 0.75/1/1、关节驱动 stiffness/damping = 100/1、手指摩擦 1.4、镊子弹簧 5；它们不是官方皮肤柔顺接触的 280/20 参数。近块撤离的 0.01 m/s 是加速度平滑前的目标，不是硬限速。通用 `scripts/run_pick_place.py` 保留原 XHand 默认配置。

记录保存在新目录中：

| 文件 | 内容 |
|---|---|
| `tacmap.npz` / `tacmap.json` | 官方输出、有效域、时间戳、实际传感器和目标位姿、调用计数、来源哈希 |
| `tactile.npz` / `tactile.json` | 独立的 480 Hz PhysX 法向力、摩擦力和接触点；不用于生成 TacMap 灰度 |
| `trajectory.npz`、`task.json`、`launch.json`、`source/` | 任务轨迹、配置、启动参数及冻结源码 |
| `evaluation.json`、各 `*_validation.json`、`independent_tacmap_task_geometry.json` | 任务、传感器和抽样解析几何验收 |

[当前主运行目录](runs/sharpa_w01_official_tacmap_matched_v2_video)保留上述数据及原生相机录像。重新生成媒体使用 [视频合成脚本](scripts/compose_sharpa_tacmap_video.py)和[响应曲线脚本](scripts/plot_sharpa_tacmap_response.py)；它们读取已有结果，不替代物理仿真。

上面的新运行完成后，在项目根目录执行以下检查。这里使用已配置分析依赖的 `.venv-human`；各检查把报告写入**新运行目录**，不覆盖冻结结果。命令参数已与各脚本的 `--help` 核对。

```bash
set -e
sharpa_run=runs/sharpa_w01_tacmap_new
sharpa_python=.venv-human/bin/python
sharpa_controls=runs/sharpa_official_tacmap_controls_matched_v1

"$sharpa_python" analysis/validate_pick_place.py "$sharpa_run" --geometry
"$sharpa_python" analysis/validate_sharpa_tactile.py "$sharpa_run" \
  --require-video --require-task-success
"$sharpa_python" analysis/validate_sharpa_tacmap.py "$sharpa_run" \
  --require-video --require-task-success --controls "$sharpa_controls"
"$sharpa_python" analysis/audit_sharpa_tacmap_task_geometry.py "$sharpa_run"
"$sharpa_python" analysis/audit_sharpa_tacmap_force_alignment.py "$sharpa_run"
```

全部检查通过后，用下面的命令合成新视频、响应曲线和压入对照图；输出文件也须使用新名称。响应曲线同时读取原版地图基线的 `tacmap.npz`、`tactile.npz`、`trajectory.npz` 和验证报告；视频读取新运行的 `overview.mp4`、`task.mp4`。两种媒体脚本都需要 `assets/fonts/NotoSansCJKsc-Regular.otf`。

```bash
set -e
sharpa_run=runs/sharpa_w01_tacmap_new
sharpa_python=.venv-human/bin/python

"$sharpa_python" scripts/compose_sharpa_tacmap_video.py "$sharpa_run" \
  --output outputs/sharpa_w01_tacmap_new.mp4 \
  --require-validated --fps 30 --max-mb 18 \
  --contact-note '当前 W01 CAD 派生表面；刚性接触；几何灰度非压力'
"$sharpa_python" scripts/plot_sharpa_tacmap_response.py "$sharpa_run" \
  runs/sharpa_w01_official_tacmap_v1_video \
  --output outputs/sharpa_w01_tacmap_new_response.png
"$sharpa_python" scripts/compose_sharpa_tacmap_controls.py \
  runs/sharpa_official_tacmap_controls_matched_v1 \
  --output outputs/sharpa_w01_tacmap_new_controls.png
```

## 已完成的结果

| 范围 | 结果 | 证据与边界 |
|---|---|---|
| 当前官方 TacMap + 匹配表面，固定案例 | **1/1 成功**；25.633 s；最高抬升 84.211 mm；落点误差 1.000 mm | [主运行评估](runs/sharpa_w01_official_tacmap_matched_v2_video/evaluation.json)；仅单例演示 |
| 同一 v6d 姿态的此前新随机测试，seed 20261023 | **6/8**；成功落点误差 0.278–1.260 mm | [逐例独立验收](runs/sharpa_w01_natural_v6d_eval_seed20261023/independent_gate_audit.json)；两失败分别在抬升和搬运阶段，均发生在释放前 |
| 同一姿态的旧坐标回归 | **8/8**；落点误差 0.406–1.008 mm | [回归验收](runs/sharpa_w01_natural_v6d_regression/validation.json)；开发案例，不是新独立样本 |

官方 TacMap 接入后**没有重新进行随机评估**。单例轨迹与原版地图对照逐字节相同，说明新增观测和派生地图没有改变这一回合的物理运动；不能据此扩大成功率结论。[独立验收汇总](runs/sharpa_w01_official_tacmap_matched_v2_video/independent_validation_summary.json)记录了轨迹哈希与检查范围。

主运行五指各 770 帧，顺序为拇指、食指、中指、无名指、小指：

| 指标 | 拇指 | 食指 | 中指 | 无名指 | 小指 |
|---|---:|---:|---:|---:|---:|
| 灰度峰值，范围 0–255 | 27 | 5 | 37 | 0 | 0 |
| 有非零值的帧数 | 770 | 447 | 165 | 0 | 0 |

[288 项 TacMap 检查](runs/sharpa_w01_official_tacmap_matched_v2_video/tacmap_validation.json)、[任务判据重构](runs/sharpa_w01_official_tacmap_matched_v2_video/validation.json)、[PhysX 力与同步检查](runs/sharpa_w01_official_tacmap_matched_v2_video/tactile_validation.json)通过。另用独立 NumPy 几何实现复核了 24 个实际时刻、共 120 个指帧，最大像素误差 1 码；这是抽样复核，未独立重算全部帧。[解析审计](runs/sharpa_w01_official_tacmap_matched_v2_video/independent_tacmap_task_geometry.json)保留选择时刻和误差。

## 算法、资产与表面地图来源

实际调用未修改的官方 `SharpaTacmap` / `SharpaTacmapCfg`，来源为 [sharpa-tacmap](https://github.com/sharpa-robotics/sharpa-tacmap)，固定 commit `e4f8827407e15625368ef0b9cf5a72c44304e16b`。算法从表面采样点沿法线发射射线，再用反向射线判断几何穿入，随后执行官方非线性量化和 9×9 高斯模糊。输出为五指各 **240×240 的 uint8 图**，随视频约 30 Hz 保存；射线目标只有镊柄和左右镊臂，不包含桌面或其他手指。

[适配器](src/sharpa_tacmap.py)负责真实传感器、动态目标和数据记录。[兼容层](src/sharpa_tacmap_compat.py)在 Lab 2.2.1 进程中补充缺失的多网格动态射线 API，使用 [Isaac Lab v2.3.2 来源代码及许可证](src/sharpa_tacmap_vendor/isaaclab_232/provenance.json)，固定 commit `37ddf626871758333d6ed89cf64ad702aef127d0`；没有覆盖系统 Lab 或修改官方 TacMap 算法。**2.3.2 是本项目兼容代码来源，不是 Sharpa 推荐版本。** [官方 TacMap README](assets/sharpa_tactile/README.md)没有明确 Sim/Lab 版本要求；另一个 [Sharpa RL 示例](https://github.com/sharpa-robotics/sharpa-rl-lab)声明测试过 Lab 2.2.0/2.3.0，不能外推为 TacMap 兼容性保证。

原版地图与 TacMap 仓库配套的旧 USD 皮肤较吻合，但与当前 W01 官方 STL 有差异。原版地图完整夹放对照五指全零；解析几何核查确认地图点未进入工具，并非仅凭“接触力非零”推测传感器故障。[表面版本比较](analysis/sharpa_tacmap/surface_alignment.json)与[原版全轨迹解析](analysis/sharpa_tacmap/natural_v6d_geometry.json)保留距离证据；[原版压入对照](runs/sharpa_official_tacmap_controls_v4/controls_validation.json)验证了传感器确实能响应规定的几何压入。

当前使用 [W01 匹配表面 v2](assets/sharpa_tacmap_w01_matched_v2/manifest.json)：由[几何生成脚本](analysis/sharpa_tacmap/build_w01_matched_maps_v2.py)把原 240×240 索引对应到当前官方 W01 三角网格最近表面，并插值法线。固定规则拒绝位移超过 2.5 mm 或法线角差超过 60° 的对应；保留拇指原有效点的 **78.34%**、四指模型的 **93.01%**。没有人为外扩，也没有根据接触力或任务轨迹拟合。

这是**派生 CAD 几何地图，不是官方原版 npy，也不是硬件 UV/像素标定**；被拒绝区域保持无效。[独立表面审计](analysis/sharpa_tacmap/matched_v2_independent_surface_audit.json)和[匹配表面的八例压入校验](runs/sharpa_official_tacmap_controls_matched_v1/controls_validation.json)均保留。原版与派生地图的对照使用相同官方算法，不能混用输入后宣称校验一致。

## 尚未对齐的物理设置

官方 [env_cfg.py](assets/sharpa_tactile/env_cfg.py) 是受控压入示例，压头为运动学刚体，`env.py` 每控制步直接设置其位姿；当前任务则让镊子和方块自由参与动力学。下面列的是**示例实际设置与当前任务差异**，不是通用推荐参数：

| 项目 | 官方压入示例 | 当前 matched_v2 任务 |
|---|---|---|
| 皮肤接触模型 | 刚体柔顺接触，stiffness/damping = **280/20** | **刚性接触**，未启用该柔顺材质 |
| 手 contact / rest offset | **+0.2 / −0.1 mm** | **+0.2 / 0 mm** |
| 对侧物体 contact / rest offset | 压头 **+0.1 / −0.5 mm** | 镊子 **+0.2 / 0 mm** |
| 物理步长、控制降采样 | 1/240 s、12，即 20 Hz 控制 | 1/480 s、8，即 60 Hz 控制 |
| 场景重力 | Z = −0.05，手与压头禁用重力 | 标准重力；动态镊子和方块受重力 |
| 接触摩擦 | 示例未显式指定；按 Lab v2.3.2 spawner 会继承静/动 0.5、`average` | 手 1.4、镊子 0.5，`max` 组合 |

当前接触偏置以[最终 USD](runs/sharpa_w01_official_tacmap_matched_v2_video/final_scene.usda)及[归档运行入口](runs/sharpa_w01_official_tacmap_matched_v2_video/source/run_pick_place.py)为准：入口显式传入 `contact_offset=.0002`，覆盖场景类的 `.0003` 默认值；不能把未使用的默认值当成实验参数。

官方手 USD 的 26 个碰撞网格均为 `convexHull`，两种压头为 `convexDecomposition`；柔顺材质和偏置在示例配置中设置。Lab 官方的 `UsdFileWithCompliantContactCfg` 会创建刚体柔顺接触材料，**不等于 FEM 皮肤**。先前只试过皮肤 280/20 和手侧负 rest offset 的短 probe，未完成双侧偏置、摩擦、模型和压入任务的整体对齐；它不能作为“完整官方物理配置失败”或当前参数已标定的证据。

当前 CAD 源表面、PhysX 碰撞代理和射线地图仍存在可测差异：中指有 **14 个非零 TacMap 帧**，对应各帧区间的全部 PhysX 子步法向/摩擦载荷均不超过 1e−9 N，帧端点为 21.0833–21.5167 s。[逐帧力—图对应](runs/sharpa_w01_official_tacmap_matched_v2_video/tacmap_force_alignment.json)保留这些记录。非零灰度不能自动证明当时受力或承托。

同一物理轨迹的源网格审计还记录了中指弹性体/右镊臂的 **329 个局部交叠控制状态、最大 0.263 mm**；299 个状态有同时刻求解器接触，30 个没有，严格接触匹配审查为 `false`。其他非目标手/工具交叠和非机制自碰为 0。不能把传感器检查通过改写为“全程零穿入”。[接触几何审查](runs/sharpa_w01_natural_v6d_video/contact_geometry_review.json)及[无接触状态复核](runs/sharpa_w01_natural_v6d_video/unmatched_middle_source_detail.json)为这项已知限制保留原始证据。

## 食指灰度较低的诊断

[低灰度诊断](analysis/sharpa_tacmap/index_low_gray_diagnosis.json)使用已有真实位姿做独立几何计算，没有调整物理参数、法线偏置或每指显示增益。结果主要来自**浅且稀疏的法线方向穿入**，量化与模糊进一步降低显示值：

- 搬运代表帧 462：食指原始法线穿入峰值 **11.24 µm**，拇指 **93.72 µm**；食指只有 17 个原始正值采样点，量化后峰值为 **2**，再由模糊降为 **1**。
- 食指峰值/最大非零范围的帧 607–608：原始峰值约 **39.7–39.9 µm**，模糊前峰值为 **7**，模糊后为 **5**。仅关闭模糊也不会得到强形变信号。
- 食指在全部 770 帧都低于拇指，但只在 162 帧低于中指；中指有 605 帧全零，其较亮信号主要出现在释放和撤离。

这些微米数是**沿采样法线的几何穿入距离**，不是最短网格穿深、实测皮肤压缩量或压力。采样点数也不是经过标定的物理接触面积。视频固定显示 0–255，数值曲线另行展示弱信号；不通过增益把低灰度伪装成更强的物理触觉。

## 后续五步改进

1. **锁定可复现环境与官方基线。** 固定 Sim/Lab、TacMap 和资产提交，先复现配套旧 USD、原地图和官方压入轨迹；分别验证包导入、动态射线和柔顺 spawner，不能仅升级版本就假定全部兼容。
2. **统一几何表面。** 在“官方配套资产 + 原地图”与“当前 W01 + 派生地图”之间明确选择，核对坐标、法线、无效域、碰撞代理和源三角面；保留 CAD 对应误差，后续再做硬件像素标定。
3. **先验证官方压入基准，再逐项检验任务候选。** 在独立压入环境中完整复现官方 280/20、双侧偏置、摩擦、步长和求解器设置，确认它们的实际绑定及输出。在自由持镊任务内保留基线，用受控对照逐项评估有物理依据的候选；不整套照搬压头参数，尤其不能把压头的 −0.5 mm rest offset 直接移到 1 mm 厚镊臂。改变接触物理后需重新检查稳定性和握持，不能只调偏置或刚度让图变亮。
4. **建立形变与受力的一致性证据。** 同步比较原始 CPD、量化前后图、PhysX 接触、源表面和碰撞代理；优先解决中指“有图无力”与食指弱响应。再用有依据的材料参数或真实压入数据检验力—位移关系，避免把刚体源交叠称为真实皮肤形变。
5. **重新验证完整任务与泛化。** 在新物理模型下重新冷起、稳定持镊、验证中指承托和释放，再跑独立随机起终点与失败回归；检查抬升、搬运跟踪/奇异位形、释放和最终落点。分别报告任务成功率、接触一致性和触觉误差，最后再考虑实机迁移。

## 文件与证据维护

运行依赖包括 [src](src)、[W01 组合机器人](assets/ur5e_sharpa_w01_right)、[官方 W01 源资产](assets/sharpa_wave/official)、[官方 TacMap 仓库](assets/sharpa_tactile)、[兼容代码与许可证](src/sharpa_tacmap_vendor)、[派生地图目录](assets/sharpa_tacmap_w01_matched_v2)，以及入口使用的姿态、快照、案例和相机配置。不得只保留演示视频而删除地图、原始数据或来源许可证。

独立检查入口为 [任务验收](analysis/validate_pick_place.py)、[官方 TacMap 验收](analysis/validate_sharpa_tacmap.py)、[PhysX 触觉验收](analysis/validate_sharpa_tactile.py)、[解析几何审计](analysis/audit_sharpa_tacmap_task_geometry.py)、[力—图对齐](analysis/audit_sharpa_tacmap_force_alignment.py)和[压入对照](analysis/sharpa_tacmap_controls.py)。清理中间文件时保留它们的最终输入、冻结源码和报告；旧开发案例的结论不能移植到新配置。

本页已根据后续差异审计更新。[冻结最终报告](reports/sharpa_w01_official_tacmap_final.json)中的 README 哈希对应[当时的文档快照](reports/cleanup_20261020/readme_before/README_SHARPA_W01_NATURAL.md)，不是本页的新哈希；原实验数据、报告和交付记录不随文档重写而改写。
