# 仿真训练数据：小步采集与验收

## 已完成的小步测试

2026-10-27 使用数据盘合并仓库执行。原计划 10 个回合，按用户要求在前几个测试通过后停止：**4 个完整回合均任务成功、数据验收通过**。另有 1 个执行中回合随队列退出，被单独归档为中断，不计入成功或完整数据；剩余回合未执行。

| Episode | 物块初始 X / Y（m） | 目标 X / Y（m） | 初始位置类型 | 最终 XY 误差 | 动作数 | 每路视频帧数 |
|---|---|---|---|---|---:|---:|
| 000 | 0.2200000 / 0.4800000 | -0.08 / 0.76 | 固定参考 | 0.434 mm | 1422 | 712 |
| 001 | 0.2095901 / 0.4892456 | -0.08 / 0.76 | 种子随机 | 0.447 mm | 1396 | 699 |
| 002 | 0.2107199 / 0.4706276 | -0.08 / 0.76 | 种子随机 | 0.466 mm | 1413 | 708 |
| 003 | 0.2080340 / 0.4657708 | -0.08 / 0.76 | 种子随机 | 0.436 mm | 1411 | 707 |

合计 5642 个控制动作、5646 个状态、双相机各 2826 帧。任务从稳定预持镊状态开始。这个小范围样本只验证采集流程与当前任务，不代表大范围随机成功率。未训练视觉模型，未做关节目标开环回放或真机执行。

本地结果目录：`simulation/runs/training_pilot_20261027/`。

- `pilot_manifest.json`：10 回合的原始计划、采样种子、随机范围与固定条件。
- `pilot_summary.json` / `.csv`：当前完成结果；`complete=false`，等待后续续采。
- `interruptions.json`、`interrupted/`：中断记录与原始部分输出，续采时不覆盖它们。
- 每个 `episode_000/` 至 `episode_003/`：完整视频、状态动作记录与验收结果。
- 本地额外检查：`verification/training_pilot_20261027/aggregate_validation.json`。

以上数据目录被 Git 忽略；本说明、采集代码及测试随代码仓库保存。

## 哪些初始条件是随机的

仅从 episode 001 起，对物块初始位置做独立均匀 XY 采样：

```text
X ∈ [0.205, 0.235] m
Y ∈ [0.465, 0.495] m
采样种子：20261027
```

episode 000 固定为已验证参考案例。每回合另设仿真种子 `20261027 + episode编号`；采样后的实际坐标通过案例文件显式传入，不依赖仅切换仿真种子。

目标点 `(-0.08, 0.76)` m、机器人 HOME/初始关节与速度、镊子预持姿态、物块尺寸和初始朝向、相机内外参、桌面、光照材质、质量摩擦、控制器与权重均固定。已对比 4 回合实际记录：初始机器人、TCP、镊子和两侧镊片状态完全一致，物块初始状态的差异仅为 XY。

固定种子与案例便于复现实验条件，不保证不同 GPU/驱动下位级一致。

## 训练动作与观测记录

开启 `--training-data` 后增加以下文件，保留原 Demo 的输出：

| 文件 | 内容 |
|---|---|
| `training_transitions.npz` | 明确对齐的状态、关节目标、控制量与诊断信息 |
| `training_schema.json` | 关节名称顺序、单位、动作语义、控制周期及图像对齐约定 |
| `training_samples.jsonl` | 验收器生成的每帧状态索引、下一动作索引及剩余动作数量 |
| `initial_conditions.json` | 请求的随机条件、实际布局与初始状态存放位置 |
| `training_validation.json` | 数据检查、任务成功与训练候选标记 |

`training_transitions.npz` 主要字段：

| 字段 | 意义 |
|---|---|
| `observation_qpos_rad` / `observation_qvel_rad_s` | 18 维实际关节位置和速度，状态输入 |
| `action_joint_position_target_rad` | 18 维绝对关节位置目标，主要训练动作标签 |
| `action_actuator_target_buffer_rad` | 实际设置到仿真 API 的位置目标缓冲区，用于独立核对 |
| `observation_T_world_tcp` | TCP 世界位姿，4×4 列向量变换 |
| `observation_tool_state_w` / `left_jaw_state_w` / `right_jaw_state_w` | 镊柄、左右镊片状态；后两个字段同样带 `observation_` 前缀 |
| `action_nominal_control` | 几何控制器的 6 维归一化控制量 |
| `action_policy_residual` | 6 维 PPO 残差 |
| `action_applied_control` | 缩放合成、限幅后的 6 维控制量 |
| `observation_block_state_w` / `observation_phase` | 物块真实状态与阶段，作为诊断信息 |
| `action_jaw_force_N` / `action_retained_error_m` / `action_face_pinch_ok` / `action_lift_m` | 接触、持镊及抬升诊断 |
| `action_done` / `action_success` / `action_failure` / `action_phase_timeout` / `action_time_limit` | 当前转移的结束状态与原因 |
| `observation_time_s` / `frame_observation_index` | 控制状态时间与双相机视频帧到状态的映射 |

关节顺序以 schema 中 `joint_names` 为准，包含 6 个机械臂和 12 个手部关节。root state 顺序为位置 xyz、四元数 wxyz、线速度 xyz、角速度 xyz。机器人实际状态与命令目标分别保存，不能将两者混用。

### 时间对应

```text
observation[t] -- action[t] --> observation[t+1]
```

T 个动作对应 T+1 个状态，控制频率 60 Hz。视频包含初始状态，随后每 2 个控制步记录一次，并补上终止状态；编码名义帧率为 30 FPS，末尾间隔可能只有 1 个控制步，精确时间以 `camera_frames.jsonl` 为准。

训练时使用 `training_transitions.npz` 的 `frame_observation_index` 或 `training_samples.jsonl` 对齐。终止画面的 `next_action_index=null`，不能配上上一动作。`trajectory.npz` 为保留的旧分析格式，其 `render_control_steps` 不含新增的初始画面，训练模式下不要用该旧字段直接索引全部视频帧。

建议视觉策略输入先使用双相机 RGB、关节位置和速度；物块真实状态、阶段和接触信息保留用于筛选与分析。

## 如何继续运行

在现有已配置服务器上，从根目录执行。续采不会重跑已完成的 000–003，下一回合为 004，使用原计划的种子与坐标：

```bash
cd /cpfs/user/xiaoqihuang/tweezers-sim2real

# 完成原计划中剩余的 6 回合。
bash scripts/run_training_pilot.sh \
  --out simulation/runs/training_pilot_20261027 \
  --resume
```

只再运行 2 个新回合，然后正常停止：

```bash
bash scripts/run_training_pilot.sh \
  --out simulation/runs/training_pilot_20261027 \
  --resume \
  --max-new-episodes 2
```

另开一批 10 回合，使用新的采样种子：

```bash
bash scripts/run_training_pilot.sh \
  --out "simulation/runs/training_pilot_$(date -u +%Y%m%dT%H%M%SZ)" \
  --episodes 10 \
  --sampling-seed 20261028 \
  --start-radius-m 0.015
```

`--out` 的相对路径以仓库根目录为基准，新批次目录必须不存在。`--resume` 使用原清单，不允许改变回合数、种子或采样半径；已记录的失败回合也会保留并跳过。发现未登记的部分目录时会停止，避免覆盖数据。

批量入口默认开启训练记录并跳过约 90 MiB/回合的展开场景导出。每回合日志为批次根目录的 `episode_NNN.log`，可另开终端用 `tail -f` 查看；单回合在当前服务器约需 6 分钟。仅继续少量测试时使用 `--max-new-episodes`，无需靠中途终止进程来停止队列。

## 复查与测试

以下离线命令使用现有仿真 Python，无需启动 Isaac Sim：

```bash
/mnt/workspace/tweezers-software/xhand-sim/bin/python \
  scripts/validate_training_episode.py \
  simulation/runs/training_pilot_20261027/episode_000

/mnt/workspace/tweezers-software/xhand-sim/bin/python \
  -m unittest discover -s simulation/tests -v
```

已通过 7 项测试，覆盖错一控制步、缺失初始帧、时间错位、用测量位置冒充命令、控制量合成错误和失败样本过滤。4 个完整回合均验证目标命令、状态/动作/视频同步、全部视频帧解码、固定相机安装关系、来源哈希和关节目标软限位。

另已实际加载前 3 回合各 3 个训练样本，得到双相机图像 `(2,3,480,640)`、关节位置与速度 `(36,)`、连续动作块 `(32,18)`。这是数据加载检查，正式训练格式转换、模型训练和学习策略闭环评估仍需后续进行。
