# 预持镊子随机采集：5 回合测试

本入口始终从手中已持镊子开始，不包含桌面取镊。每个回合都采样，包括 episode_000；没有固定的参考回合。

## 默认采样范围

| 项目 | 默认范围 / 方法 |
|---|---|
| 物块初始 XY | X ∈ [0.20,0.24] m；Y ∈ [0.46,0.50] m |
| 目标 XY | X ∈ [-0.10,-0.06] m；Y ∈ [0.74,0.78] m |
| 起终点最小距离 | 0.25 m，并检查桌面边界余量 |
| HOME XY | 相对 (0,0.48) 每轴 ±5 mm |
| HOME 高度 | 镊尖初始高度相对参考值 ±3 mm |
| 工具偏航 | 相对参考布局的 90°，再扰动 ±1° |
| 机械臂初始关节 | 对整体持镊姿态做上述变化后，由 IK 求解；使用现有桌面间隙检查 |
| 手部位置目标 | 12 个手部关节各 ±0.001 rad，受原有手部边界约束 |
| 初始物理稳定 | 保持位置目标，先执行至少 0.35 s 物理仿真，通过持镊检查后开始记录 |
| 腕部相机外参 | 相机局部坐标各轴平移 ±1.5 mm；旋转向量各分量 ±0.5° |
| 全局相机外参 | 相机局部坐标各轴平移 ±5 mm；旋转向量各分量 ±0.5° |

平移和偏航改变整个预持镊状态，保持手、镊柄、左右镊片的相对几何关系。手部扰动施加在位置目标上，由接触物理产生实际关节变化；不是独立瞬移手指。手部目标偏移在该回合中保持，避免开始动作后立即恢复另一种持握配置。部分偏移可能被关节边界截断，文件会同时保存请求偏移和有效偏移。

腕部扰动作用于相机相对 TCP/连杆的安装变换，随连杆运动；全局扰动作用于相机相对基座的安装变换。每回合内部安装关系固定，保留原有内参。旋转范围指旋转向量的各分量，不是总旋转角范围。

当前不随机光照、质量、摩擦、物块尺寸和初始朝向。目标以绿色可视边框表示，后续真机应使用相同目标信息约定。

配置文件：[randomized_collection_v1.json](../simulation/configs/randomized_collection_v1.json)。这是保守的首版范围，5 个完整回合的任务成功率需要运行后再统计。

## 执行已准备的 5 回合计划

默认种子为 20261027，每回合使用 seed + episode 编号，并保存全部采样结果。当前已准备好以下批次的输入，尚未运行其中的完整回合：

```bash
cd /cpfs/user/xiaoqihuang/tweezers-sim2real

bash scripts/run_randomized_collection.sh \
  --out simulation/runs/randomized_grasp_test_5ep_20261027 \
  --resume
```

完整数据最终存放在：

```text
/cpfs/user/xiaoqihuang/tweezers-sim2real/simulation/runs/randomized_grasp_test_5ep_20261027/
```

只先执行 1 回合然后正常停止：

```bash
bash scripts/run_randomized_collection.sh \
  --out simulation/runs/randomized_grasp_test_5ep_20261027 \
  --resume --max-new-episodes 1
```

以后用相同 `--resume` 命令继续，已登记的完整尝试（包括失败尝试）会跳过。不会自动重采直到凑够成功回合；5 回合指 5 次计划尝试。未登记的部分输出不会被覆盖，需要先保存到独立目录再重试。

## 另建新批次

```bash
bash scripts/run_randomized_collection.sh \
  --out "simulation/runs/randomized_grasp_$(date -u +%Y%m%dT%H%M%SZ)" \
  --episodes 5 --seed 20261028
```

只生成采样清单、不启动仿真：

```bash
bash scripts/run_randomized_collection.sh \
  --out "simulation/runs/randomized_plan_$(date -u +%Y%m%dT%H%M%SZ)" \
  --episodes 5 --seed 20261028 --plan-only
```

相对输出路径以仓库根目录为基准。新批次目录必须不存在。续采检查输入和关键代码/模型哈希，避免采样文件或代码改变后混入同一批次；改参数或代码后应建立新批次。

## 输出结构

```text
randomized_grasp_test_5ep_20261027/
├── collection_manifest.json       采样范围、全部回合参数、输入/代码/模型哈希
├── collection_summary.json        已执行回合的任务与数据检查结果
├── collection_summary.csv         便于汇报的汇总表
├── inputs/
│   ├── episode_000/
│   │   ├── conditions.json        起终点、HOME、手部偏移和相机扰动
│   │   ├── case.json              传给任务的物块起点和目标
│   │   └── layout.json            传给场景的 HOME 和布局
│   └── ...
├── episode_000.log                当前回合运行日志
├── episode_000/
│   ├── initial_conditions.json    初始稳定检查、实际关节/工具/物块状态
│   ├── camera_rig_effective.json  本回合真正使用的相机外参
│   ├── wrist.mp4 / global.mp4
│   ├── training_transitions.npz   状态、18 维关节目标与诊断数据
│   ├── training_schema.json       关节顺序、单位、时间与动作定义
│   ├── training_samples.jsonl     帧到状态/后续动作索引
│   ├── training_validation.json   数据检查和训练候选判定
│   ├── trajectory.npz            兼容旧分析工具的轨迹
│   ├── camera_frames.jsonl       每帧时间和实际 TCP/相机位姿
│   ├── evaluation.json           任务成功判定
│   └── source/ ...               原有配置、来源与截图
└── episode_001/ ... episode_004/
```

每回合通过持镊初始检查后才开始录像和训练时间计数。初始稳定检查失败会保存原因，不作为有效轨迹。完整回合结束后，自动解码双路视频并检查状态动作对齐、相机安装关系、命令一致性和目标软限位。失败任务与数据损坏分别记录；正式训练优先使用 `training_candidate=true` 的回合。

数据在 `simulation/runs/` 下，被 Git 忽略。代码上传 GitHub不等于备份采集数据。

## 已执行的验证范围

- 13 项离线测试通过，包含种子复现、采样边界与距离、相机变换链保持、错误数据与时间错位识别等。
- 已实际运行独立的 `simulation/runs/randomized_grasp_smoke_20261027/`：一个 0.5 s 短回合，前置物理稳定检查通过，初始机械臂和手部实际关节均发生变化，两路外参也发生变化；数据验收与视频解码通过。
- 短回合因设置的时间上限结束，未完成夹取，不能记作任务成功。上面的正式 5 回合计划使用 36 s 上限，由用户执行。

训练字段详细定义见 [TRAINING_DATA_PILOT.md](TRAINING_DATA_PILOT.md)。原有固定初始条件的 Demo 仍可用统一入口运行。
