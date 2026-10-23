# 灵巧手镊子操作 Sim-to-Real 项目

**项目目标：** 通过 Isaac Sim 仿真数据与真实灵巧手数据混合训练 VLA / WAM，实现真机自主使用镊子夹取物块。

**候选模型：** π₀ / π₀\.₅ / WAM

**最后更新：** 2026-10-23。

**当前进度：** UR5e + XHand 镊子抓放仿真 Demo 已跑通；D405 腕部相机、D435 全局相机已完成真机标定和仿真接入；已调整仿真桌面与任务布局，并验证双相机录制及单回合轨迹记录。真机侧已完成相机标定数据采集，尚未记录完成自主抓放或操作示教数据集。

**进度记录口径：** 后续按实际完成情况更新本文件、日期及结果路径；优先记录可核对的代码、输出和指标。单回合成功与批量评估分别记录，仿真结果与真机结果分别记录；尚未开展的阶段保留原有占位，不提前填写成果。当前使用的残差 PPO 策略用于生成仿真抓放动作，不代表已完成 VLA / WAM 训练。

**本地项目根目录：** `/mnt/workspace/tweezers-sim2real`（合并仓库；CPFS 配额恢复后可整体迁移）。以下链接均相对本进度文档所在目录。

---

**仓库整理（2026-10-23）：** 已将真机标定与仿真项目整合为本仓库，入口见 [README.md](README.md)。标定源码在 `calibration/`，仿真源码在 `simulation/`，冻结策略和抓握快照在 `models/`；原始采集数据、历史结果和冻结确认包已复制保留，但按目录排除在 Git 之外。原两个目录保留。统一入口迁移后的验证记录见 `verification/merge_validation.json`。

## Phase 1：Isaac Sim 仿真环境搭建

**阶段任务：**

- 完成灵巧手、镊子及物块的仿真建模和物理参数配置。

- 实现灵巧手控制镊子完成夹取、移动及释放任务。

**阶段结果：**

- 完成情况：**基础仿真抓放与已标定双相机 Demo 已完成。**

  - 已配置 Python 3.11、PyTorch 2.7.0+cu128、Isaac Sim 5.0.0、Isaac Lab 2.2.1，空场景及完整任务均已运行。
  - 已加载 UR5e + XHand 机器人资产，配置镊子、物块、桌面及接触物理参数，完成接近、下降、夹紧、抬升、搬运、释放和撤离的完整流程。
  - 当前控制方案为“几何阶段控制器 + 已训练的残差 PPO”，评估时使用确定性策略输出。沿用 checkpoint：`models/policies/xhand_pick_place_ppo.pt`；checkpoint 记录为 25 次迭代、153600 个采样 transition，本次工作主要完成部署、相机接入和布局调整。
  - 已接入 D405 腕部相机与 D435 全局相机，使用已确认的内外参和 rectified 图像模式。腕部相机跟随 `wrist_3_link`，全局相机相对机器人基座固定。
  - 已将桌面长边调整到世界 Y 轴：尺寸 X×Y×Z 为 `0.80×1.10×0.04 m`，中心 XY 为 `(0, 0.35)`，台面高度 `0.72 m`；物块初始 XY 为 `(0.22, 0.48)`，目标 XY 为 `(-0.08, 0.76)`。
  - HOME 对应的镊尖 XY 调整为 `(0, 0.48)`，稳定抓握整体绕世界 Z 轴旋转 90°，通过 reset IK 重算机械臂起始关节角。该布局为便于视觉采集的近似布局，未按真实桌面尺寸精确建模。

- 仿真 Demo：

  - 最新结果目录：[merged_repo_demo_20261023](simulation/runs/merged_repo_demo_20261023/)。
  - [全局视频](simulation/runs/merged_repo_demo_20261023/global.mp4)、[腕部视频](simulation/runs/merged_repo_demo_20261023/wrist.mp4)、[全局首帧](simulation/runs/merged_repo_demo_20261023/global_first.png)。
  - seed=0 的指定案例成功 **1/1**，最终 XY 误差约 **0.434 mm**，最大抬升约 **8.44 cm**，仿真时长约 **23.70 s**。这是该单案例的结果，尚不能作为随机场景总体成功率或真机精度。
  - 两路视频均为 `640×480`、30 FPS、H.264/yuv420p，各 712 帧，已逐帧解码检查；相机位姿与时间记录检查通过，执行来源边界检查一致。见 [demo_verification.json](simulation/runs/merged_repo_demo_20261023/demo_verification.json)。

- 问题以及解决方案：

  - **依赖安装与存储：** 安装过程中遇到下载源访问失败及 Miniforge 解包失败，后续改用 Miniconda 完成环境安装；现有仿真环境位于 `/mnt/workspace/tweezers-software/xhand-sim`。
  - **图形库与 GPU 初始化：** 初次启动缺少 GLX / Xt / GLU 等系统动态库；补齐相关依赖后，继续排查 Vulkan 仅识别 CPU 渲染器的问题。
  - **驱动库不匹配：** 新实例内核驱动为 570.153.02，但曾加载 CUDA compat 提供的 580.95.05 `libcuda`，出现 `cuInit=804` 等错误；通过匹配的 570 用户态驱动库、`LD_LIBRARY_PATH`、NVIDIA Vulkan ICD 与 EGL vendor 配置恢复 CUDA / Vulkan。已将当前实例所需配置封装到 [setup_xhand.sh](env/setup_sim.sh)。该脚本选择已有库，不负责安装或更换服务器内核驱动。
  - **空场景启动后无输出：** `--headless` 下出现 `Setup complete` 后，空场景持续执行仿真循环，终端不再持续打印属于正常行为。
  - **相机投影与坐标转换：** 已核对 base/base_link、TCP/link、光学轴与四元数约定；针对原生相机方形像素限制，使用已验证的针孔渲染与按目标 K 重采样方案，通过 `rig.rgb()` 获取 rectified 图像。
  - **global 视角像侧拍：** 标定相机已接入，但原桌面与作业区域主要沿 X 方向展开；调整桌面、HOME、工具初始朝向及抓放点后，global 视角已改善。相机标定矩阵继续沿用已确认结果。

**当前复跑命令：**

```bash
cd /mnt/workspace/tweezers-sim2real/simulation
RUN_DIR="runs/xhand_calibrated_y_table_$(date -u +%Y%m%dT%H%M%S_%N)"
bash scripts/run_calibrated_demo.sh "$RUN_DIR"
```

脚本会加载环境配置。布局参数见 [calibrated_demo_y_table.json](simulation/configs/calibrated_demo_y_table.json)，物块初始与目标位置见 [calibrated_demo_y_case.json](simulation/configs/calibrated_demo_y_case.json)，使用说明见 [README_CALIBRATED_DEMO.md](simulation/README_CALIBRATED_DEMO.md)。

## Phase 2：仿真数据采集

**阶段任务：**

- 实现仿真环境中的操作轨迹生成、数据采集与回放。

- 通过场景随机化采集不同条件下的示教数据。

**阶段结果：**

- 完成情况：**单回合轨迹和双视角录制能力已验证，尚未完成批量数据集采集。**

  - 当前 Demo 可运行策略生成完整抓放轨迹，并同步记录腕部与全局 RGB、关节状态、物体状态、动作、任务阶段及成功标志。
  - `trajectory.npz` 记录 60 Hz 控制步数据；视频以 30 FPS 记录，`render_control_steps` 提供视频帧对应的控制步索引，`camera_frames.jsonl` 另存每帧精确仿真时间和位姿。
  - 控制器已有初始/目标 XY 随机采样功能，当前已验收的新布局使用固定案例。场景随机化批量采集及数据集回放验收尚未记录完成。

- 数据集规模：当前新布局的已验证样例为 **1 个成功回合、1422 个控制步、双视角各 712 帧**；两路图像对应同一回合，不计为两个独立 episode。尚无正式批量数据集规模。

- 数据存储路径：[tweezers/runs/merged_repo_demo_20261023](simulation/runs/merged_repo_demo_20261023/)。

  - `wrist.mp4`、`global.mp4`：双视角视频；`*_first.png`、`*_last.png`：首尾截图。
  - `trajectory.npz`：18 维关节状态、镊子和物块状态、策略残差、实际施加动作、阶段、成功标志等。
  - `camera_frames.jsonl`：帧索引、仿真时间、关节位置、TCP 及两路光学相机的世界位姿。
  - `camera_manifest.json`、`scene_layout.json`：相机与布局配置；`evaluation.json`、`demo_verification.json`：任务结果和输出检查。
  - `launch.json`、`source/`、`execution_source_provenance.json`：参数、来源存档及边界哈希检查。

- 存在问题：当前输出仍是 Demo / 分析格式；大规模采集、正式数据集格式转换及训练输入验收尚未完成。

## Phase 3：真机实验与数据采集

**阶段任务：**

- 完成真实灵巧手的控制、遥操作及相机数据同步。

- 采集真实环境下使用镊子夹取物块的示教轨迹。

**阶段结果：**

- 完成情况：**已完成真机相机标定及验证数据采集；操作示教采集尚未记录完成。**

  - 真机平台为 UR5e，腕部相机 Intel RealSense D405，全局相机 Intel RealSense D435；采集主机为 Ubuntu 22.04.5 LTS。
  - 已搭建相机标定 pipeline，完成两路 RGB 最终 v3 内参、D405 手眼外参 `T_TCP_Cw`、D435 全局外参 `T_B_Cg` 的求解与结果回传。
  - 本次真实激活 TCP 偏置确认为 `[0,0,0,0,0,0]`。
  - 两路真实独立验证均通过：腕部 10 张验证图像，重投影 RMS 平均约 1.042 px、最大约 1.629 px；全局 6 张验证图像，RMS 平均约 0.339 px、最大约 0.380 px。
  - 标定采集记录了腕部图像对应的静止机器人关节角与 TCP 位姿；这不代表两路 RGB 与运动状态的动态同步采集已完成。

- 数据集规模：回传包中包含腕部 10 个验证样本、全局 6 个验证样本，属于标定验证数据。操作示教轨迹数据集规模尚未记录；内参及手眼训练原图按交接说明留在采集机归档。

- 真机 Demo：当前确认完成的是标定采集与验证，尚无本阶段自主使用镊子抓放的已验收 Demo。

- 存在问题：

  - 真实灵巧手控制、遥操作、双相机动态同步和抓放示教采集尚未记录完成。
  - 双相机对同一目标的联合交叉验证尚未采集；**根据当前项目取舍，暂缓此项，不作为继续仿真视觉采集的前置条件**。这与已经完成的单相机验证、真实/仿真 TCP 对比是不同检查。
  - 当前标定结果依赖本次相机安装、图像模式与 TCP 配置；实际变更后需复查相关参数是否仍适用。

**标定资料：** 原始交接包 [xhand_calibration_handoff_20261023.zip](archives/xhand_calibration_handoff_20261023.zip)；仿真侧确认材料见 [calibration_confirmed_20261023](archives/calibration_confirmed_20261023/)。

## Phase 4：Sim/Real 数据对齐

**阶段任务：**

- 统一仿真与真实数据的观测空间、动作空间及数据格式。

- 构建支持 Sim/Real 混合训练的数据处理 Pipeline。

**阶段结果：**

- 完成情况：**已完成相机几何参数与仿真坐标链对齐；训练数据层面的统一尚未完成。**

  - 已核对实际加载的 UR5e + XHand USD / URDF、固定连杆链、单位和运行时姿态 API，确认 `reference_link=wrist_3_link`、`T_link_tcp=I`，使用连杆原点的 `body_link_pos_w/body_link_quat_w`，四元数为 wxyz。
  - 当前 env_0 的 `T_world_base` 为绕 Z 轴 180° 旋转加 `[0,0,0.72] m` 平移，已包含 base_link→UR 控制器 base 的固定变换。
  - 已用 10 组真实关节状态进行仿真运行时复查：真实/仿真 TCP 平移差平均约 0.782 mm、最大约 0.853 mm，朝向差平均约 0.216°、最大约 0.228°。
  - 已完成 5 个姿态的双相机 rectified 渲染投影检查，独立颜色校验点最大像素误差为腕部约 0.313 px、全局约 0.285 px。全局相机/板几何固定，重复渲染不计为多个独立真实验证视角。
  - 已将验证后的双相机适配器接入正常抓放 Demo，并完成近似桌面与作业区调整。

- 数据格式：真实标定数据采用 PNG + `session.json` + `samples.jsonl`；仿真 Demo 采用 MP4 / PNG + JSON / JSONL + NPZ。坐标变换统一使用列向量、米和 `T_A_B` 将 B 坐标映射到 A 的约定。两端的训练观测、动作语义、时间对齐及最终数据集格式尚未统一验收。

- 代码路径：

  - [CONFIRMATION_REPORT.md](archives/calibration_confirmed_20261023/CONFIRMATION_REPORT.md)：确认结论、真实验证与渲染指标。
  - [T_LINK_TCP_CONFIRMATION.md](docs/T_LINK_TCP_CONFIRMATION.md)：固定链、基座与 TCP 核查流程及完成记录。
  - `archives/calibration_confirmed_20261023/xhand_calibration_handoff/`：标定工具、最终 `camera_rig_isaac.json` / `world_base_isaac.json` 及仿真相机适配器。
  - [run_calibrated_demo.sh](simulation/scripts/run_calibrated_demo.sh)、[calibrated_demo_render.py](simulation/src/calibrated_demo_render.py)：标定双相机 Demo 入口与录制实现。

- 存在问题：

  - 桌面当前采用近似布局；手部、镊子和物块的尺寸、质量、摩擦等尚未完成真实参数辨识。
  - 实际使用并验证的是 rectified RGB；raw 畸变渲染、深度数据和动态跟踪尚未验收。渲染小尺寸 ChArUco 标签解码未充分通过，投影验收采用独立颜色校验点。
  - 真实到模型仍有残差，不能将渲染器内部的亚像素投影误差理解为真机绝对精度。
  - Sim/Real 操作数据统一、混合训练数据处理 pipeline 尚未完成；后续阶段暂保留原记录占位。

## Phase 5：VLA / WAM 模型训练

**阶段任务：**

- 适配 π 系列或 WAM 模型，完成灵巧手动作预测与模型训练。

- 对比 Real\-only、Sim \+ Real 混合训练及 Sim → Real 微调等方案。

**阶段结果：**

- 最佳模型：

- Checkpoint 路径：

- 存在问题：

## Phase 6：真机部署与评估

**阶段任务：**

- 将训练模型部署至真实灵巧手，实现自主闭环操作。

- 评估真实夹取任务成功率，分析 Sim/Real 混合训练的效果。

**阶段结果：**

- 最终 Demo：

- 最佳模型：

- 存在问题及改进方向：

---

## 项目最终成果

- Isaac Sim 灵巧手镊子操作仿真环境

- Sim/Real 镊子操作数据集

- VLA / WAM 混合训练 Pipeline

- 真机自主镊子夹取 Demo

- Sim\-to\-Real 实验对比与分析

**代码仓库：**

**数据存储路径：**

**最终任务成功率：**
