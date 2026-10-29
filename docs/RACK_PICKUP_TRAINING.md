# 独立取镊 PPO

本入口复用现有 `ppo.py`，随机初始化新网络，不加载或修改原夹物策略。
场景使用已确认的旋转支架和无定位槽托舌。训练只有关节位置驱动；物体状态仅在回合重置时写入。

## 固定起点版本 v2（2026-10-28）

当前任务为 `rack_pickup_fixed_start_v2`：机械臂重置到独立配置的六个固定关节角，同一个 PPO 从第一阶段开始参与移动到预抓取位姿及后续取镊。支架位置只影响目标位姿，不再用于选择初始关节角。第一阶段奖励公式、配置和验证结果见 [固定起点说明](RACK_PICKUP_FIXED_START.md)。旧 v1 检查点不能直接续训到 v2，请使用新的输出目录从随机权重开始训练。

## 物块桌面位置修正（2026-10-28）

原训练场景沿用物块与镊子相同的初始 XY，导致物块与支架底座重叠并落到底座上。现将物块位置独立为 `simulation/configs/tweezer_rack_scene.json` 的 `block_xy_m: [0.0, 0.65]`，中心出生高度由桌高加 6.1 mm 计算，重置也恢复该桌面位置。预览入口采用同一配置。原夹物场景未指定 `block_xy` 时仍使用原默认行为。

修正后的零残差完整评估成功 1/1，用时 15.633 秒；物块中心全程保持约 `(0.0, 0.65, 0.726)` m，12 mm 物块底面位于 0.72 m 桌面。两路视频各 469 帧完整解码，结果与 `block_placement_check.json` 位于 `simulation/runs/rack_pickup_table_block_20261028/`。本次评估轨迹新增 `block` 状态字段，便于核对全程位置。

本次变更更新了场景/任务源码签名，并把物块初始位置加入检查点兼容性检查。下方旧短程模型是修正前的历史验证产物，当前入口会拒绝直接加载；新训练请使用当前配置生成新检查点。下方正式训练与评估命令继续适用，但“复查本机已有短程模型”命令只适用于其原源码版本。

## 历史 v1 验证（2026-10-28）

训练、断点续训、保存模型加载和完整固定场景评估均已有运行结果。本次收尾重新核对了这些结果与当前源码的 SHA-256、检查点权重及轨迹有限值，并通过 3 项离线回归测试，没有重新启动正式长时间训练。

| 检查 | 结果 | 本地输出（相对 `simulation/runs/`） |
|---|---|---|
| 零残差基础控制器完整回合 | 成功 1/1，15.633 秒 | `rack_pickup_nominal_20261028_04/` |
| 新 PPO 短程训练 | 2 环境 × 256 步 × 4 次迭代，共 2,048 条样本；84 维观测、9 维动作；权重有更新且有限 | `rack_pickup_training_check_20261028/` |
| 断点续训 | 接至第 5 次迭代，累计 2,064 条样本 | `rack_pickup_resume_check_20261028/` |
| 已保存 PPO 确定性评估 | 成功 1/1，15.683 秒，经过全部六阶段并保存实际交接候选状态 | `rack_pickup_checkpoint_eval_20261028/` |
| 自动重置路径 | 缩短回合时限的 2 环境、256 条样本检查完成；仅验证重置与更新流程 | `rack_pickup_reset_check_20261028/` |
| 基础控制器视频 | 全景、支架近景均为 960×720、30 FPS，各 469 帧完整解码 | `rack_pickup_nominal_20261028_04/overview.mp4`、`rack_closeup.mp4` |

保存模型评估末帧的相对握姿位置误差为 **2.567 mm**，HOME 工具位置误差为 **2.377 mm**，相对握姿角度误差约 **4.50°**，开合测试间距响应为 **1.513 mm**；末帧工具/机器人与支架接触合力均为 0 N。完整 941 步轨迹中的关节、目标、工具状态、动作和奖励均为有限值。

本次核验报告：[verification.json](../verification/rack_pickup_completion_20261028/verification.json)；离线测试记录：[tests.txt](../verification/rack_pickup_completion_20261028/tests.txt)。这些证据与运行目录为本机数据，受 Git 忽略规则管理。

基础控制器视频来自较早的源码版本；随后增加了配置校验、源码签名、日志字段，并将同一掉落判据提取为函数。短程训练、续训和保存模型评估所归档的源码均与本次核验前的源码一致。收尾时仅调整 Python 入口的参数注册顺序，修复 Isaac Lab 预解析导致 `--help` 错误要求 `--out` 的问题；帮助、缺少必填参数、非法环境数量三项命令行检查通过，任务及 PPO 逻辑未变。视频展示的是零残差控制器，不是训练后 PPO 的录像。

以上证明训练与评估入口可运行，且当前固定案例可完成取镊。由于基础控制器本身也能成功，不能据此宣称 PPO 已带来性能提升；尚未完成随机场景总体评估、原 PPO 交接容忍度测量或“取镊后连续夹物”的联合验证。

## 任务与控制

六阶段：固定起点移动到预抓取位姿 → 接近 → 夹持 → 上提脱架 → 移离支架并返回 HOME → 稳定与开合检查。

9 维有界残差为腕部位置目标偏移 XYZ、腕部旋转向量偏移 XYZ、闭合协同、食指支撑、中指支撑。
位置/旋转残差是相对阶段目标的有界偏移，不随控制步累计。基础腕部轨迹在指令空间做阻尼逆解；手指目标平滑并限位。
物理频率 480 Hz，控制频率 60 Hz。原 Actor–Critic 隐藏层为 256/256/128。

取镊参考由稳定快照的腕部—镊子相对变换计算。HOME 通过 `StableResetReference` 和当前 demo 布局计算，使用原流程的镊尖高度与姿态，不把预览图的出生点误当交接 HOME。
原快照只提供几何与手指参考，不作为取镊任务的持镊初始化。

夹持使用真实拇指/食指对应镊片的面法向力；支架接触采用真实 PhysX 法向合力，并对一个控制步内的物理子步取最大值。
取下之后沿托舌向外撤离（默认世界 −Y），避免横向穿过托舌。阶段 4 先完成撤离航点，再返回 HOME。

## 奖励和终止

连续奖励乘以控制周期，事件奖励只在触发时计算：

- 移动到预抓取位姿：距离/朝向的有符号进度奖励，距离/朝向误差、手指偏离张开姿态、近目标高速运动的惩罚，沿用时间成本和碰撞惩罚。第一阶段误差项为非正奖励率，减少原地等待获取正奖励的动机。
- 近距离接近：沿用位置与朝向对齐奖励，惩罚推移镊子。
- 夹持：双侧有效接触、相对握姿、相对稳定性；接触力超过上限后惩罚。
- 提起：有握持条件下的实际提起高度、握姿、稳定性。
- 移离并返回：目标接近、握姿和稳定性。
- 交接：握姿、稳定性和有方向的开合响应。
- 全程：残差幅度、残差变化、时间成本、机器人撞架；提起后持续工具撞架也惩罚。
- 阶段推进 +5；首次有效脱架 +10；交接成功 +30；物理失败 −15；超时 −5。

阶段转换要求连续满足条件，单纯经过固定时间不会宣告抓住或脱架。
进入交接后先执行小幅开、合各 0.5 秒（期间屏蔽手指残差，避免取消测试），再恢复握持并稳定保持。
候选交接要求已脱架、HOME 误差、手—工具相对位姿、手指状态、接触力与相对速度合格，且开合响应通过。
当前容差是工程初值，尚不是原 PPO 容忍度测试得到的范围。

失败记录区分：非有限状态、掉落、初始推移、失去握姿、严重撞架、机器人原点低于桌面、阶段超时、整回合超时。
后半程正常降低到 HOME 不使用“低于支架固定高度即掉落”的判据。

## 命令

以下命令从仓库根目录执行；`--out` 相对于 `simulation/`，必须是新目录。

```bash
cd /cpfs/user/xiaoqihuang/tweezers-sim2real

# 基础控制器评估并录像，不使用 PPO 残差
bash scripts/train_rack_pickup.sh \
  --out runs/rack_pickup_fixed_zero_01 \
  --num-envs 1 --evaluate zero --video --device cuda:0

# 初期训练检查：20 次迭代，共 40,960 条交互样本
bash scripts/train_rack_pickup.sh \
  --out runs/rack_pickup_fixed_pilot_01 \
  --num-envs 8 --iterations 20 --rollout-steps 256 \
  --save-every 5 --learning-rate 0.0001 --std 0.15 \
  --seed 0 --device cuda:0

# 从随机网络开始正式训练：1000 次迭代，共 2,048,000 条交互样本
bash scripts/train_rack_pickup.sh \
  --out runs/rack_pickup_fixed_train_01 \
  --num-envs 8 --iterations 1000 --rollout-steps 256 \
  --save-every 25 --learning-rate 0.0001 --std 0.15 \
  --seed 0 --device cuda:0

# 从该新技能的检查点继续训练（iterations 是新增迭代数）
bash scripts/train_rack_pickup.sh \
  --out runs/rack_pickup_fixed_train_02 \
  --resume runs/rack_pickup_fixed_train_01/latest.pt \
  --num-envs 8 --iterations 500 --rollout-steps 256 \
  --save-every 25 --device cuda:0

# 评估训练出的模型并录像
bash scripts/train_rack_pickup.sh \
  --out runs/rack_pickup_fixed_eval_01 \
  --evaluate runs/rack_pickup_fixed_train_01/latest.pt \
  --num-envs 1 --eval-episodes 3 --video --device cuda:0
```

训练默认是固定支架与独立固定关节起点，用于先建立单案例能力。多次固定场景评估不能代表随机总体成功率。
`--task-config` 可传入自定义 JSON，字段见 `RackPickupConfig`。可逐步增加 `reset_arm_noise_rad`，但更换任务配置后不能当作同配置断点续训；当前入口会拒绝不一致的任务配置。
不支持直接加载原 6 维夹物 checkpoint。续训恢复优化器学习率，`--learning-rate` 不覆盖已保存的优化器状态。

建议先运行 20 次迭代的 pilot，查看 `training.jsonl` 中的阶段占比、失败原因和最近 100 回合成功率，再单独评估 `latest.pt`。正式训练命令给出的 1,000 次迭代是初始预算，不是保证收敛的训练量。训练日志中的 `success_rate_100` 由 PPO 采样动作产生，评估默认使用确定性动作，两者不应直接混作同一个成功率。

旧 `rack_pickup_training_check_20261028/latest.pt` 属于历史版本，不能由当前 v2 入口加载。评估时请使用当前版本训练出的检查点。

查看训练日志（在仓库根目录另开终端）：

```bash
tail -f simulation/runs/rack_pickup_fixed_pilot_01/training.jsonl
```

运行入口：[scripts/train_rack_pickup.sh](../scripts/train_rack_pickup.sh)；Python 入口：[train_rack_pickup.py](../simulation/scripts/train_rack_pickup.py)；任务与奖励：[rack_pickup_task.py](../simulation/src/rack_pickup_task.py)；默认配置：[rack_pickup_train.json](../simulation/configs/rack_pickup_train.json)。

## 输出

- `task.json`：完整任务参数、动作语义、实际 HOME 与相对握姿参考。
- `ppo_config.json`：PPO 超参数。
- `training.jsonl`：损失、KL、奖励、各阶段占比、终止原因比例、最近 100 回合成功率。
- `untrained.pt`、`latest.pt`、`iteration_XXXXX.pt`：初始化和训练检查点。
- `launch.json`、`source/`、`source_check.json`：参数和源码归档及运行边界哈希检查。
- `evaluation.json`：逐回合成功、最远阶段、终止原因和时长。
- `episode_NNN.jsonl`、`.npz`：逐步奖励、状态、最终关节目标与残差。
- `overview.mp4`、`rack_closeup.mp4`：启用 `--video` 后的展示视频。
- `handoff_NNN_EEE.json`：仅成功回合保存的实际交接候选状态。

交接文件不代表已经通过原夹物 PPO 的连续验证。当前入口负责独立取镊技能的训练和验收；不自动调用原策略，也不会在交接时用快照覆盖真实抓取状态。
