# 灵巧手镊子夹取物块：Sim-to-Real

UR5e + XHand 使用预持镊子夹取、搬运和释放物块。包含 **D405 腕部 / D435 全局相机真机标定、Isaac Sim 抓放仿真、双视角与轨迹记录**，统一维护在一个 Git 仓库。

截至 2026-10-23，真机标定与仿真相机投影检查已通过；Y 轴桌面布局完成指定案例抓放 1/1。VLA / WAM 训练、真机自主操作和正式混合训练数据集尚未完成。当前控制为几何阶段控制器 + 已训练残差 PPO，任务从已经持镊开始。

## 目录与入口

```text
tweezers-sim2real/
├── calibration/           真机标定 Python 包、相机适配器、配置
│   ├── src/xhand_calib/   图像采集、内参、手眼、全局外参、验证、导出
│   ├── examples/          已验证 Isaac 相机适配器与 lens schema bootstrap
│   ├── results/           最终参数及历史求解结果；顶层 JSON 加入 Git
│   └── data/              已复制原始采集数据；本地保留，Git 忽略
├── simulation/            Isaac Sim 场景、任务、PPO、配置、历史工具
│   └── runs/              已复制历史运行以及新输出；Git 忽略
├── assets/                UR5e + XHand 的完整 URDF/USD/网格
├── models/                冻结 PPO checkpoint 与稳定预持镊快照
├── scripts/               统一运行、标定、输入检查、离线复验入口
├── env/                   仿真终端配置与机器本地配置模板
├── docs/                  坐标说明、迁移说明、参考照片
├── archives/              原确认包、交接 ZIP、校验材料；Git 忽略
├── verification/          合并后检查及验证结果；Git 忽略
├── PROJECT_PROGRESS.md    当前实际项目进度
└── MERGE_PROVENANCE.json  合并来源、版本和关键输入哈希
```

当前机器的合并仓库在 `/mnt/workspace/tweezers-sim2real`。创建时 CPFS 返回 `Disk quota exceeded`，所以先在这里完成。**此路径属于当前实例系统盘，停止/更换实例前，应把整个目录或完整压缩包保存到配额足够的持久存储。** 原两项目未删除，仿真软件安装与 Python 环境继续使用当前机器已有安装。

## 运行已标定双相机抓放 Demo

当前服务器已经安装依赖，直接执行：

合并后已用下述统一入口重跑完整回合：指定案例成功 1/1，最终 XY 误差约 0.434 mm；两路视频各 712 帧，逐帧解码通过。16 项标定测试和两路真实验证数据复算通过，详细记录见 [docs/VALIDATION.md](docs/VALIDATION.md)。

```bash
cd /mnt/workspace/tweezers-sim2real
bash scripts/run_demo.sh
```

统一脚本自动加载 `env/setup_sim.sh`。指定结果目录或追加参数：

```bash
bash scripts/run_demo.sh runs/my_demo --seed 0
# 仅检查启动与录像，不作为完整任务验收：
bash scripts/run_demo.sh runs/my_smoke --episode-seconds 0.5
```

相对输出路径以 `simulation/` 为基准；目录必须不存在。默认录制 `wrist.mp4`、`global.mp4`，并保存 `trajectory.npz`、`camera_frames.jsonl`、相机/布局配置及成功判定。两路图像为 rectified、640×480、30 FPS、H.264/yuv420p。

当前桌面 X×Y 为 `0.80×1.10 m`，长边沿 Y，中心 XY `(0,0.35)`、高度 `0.72 m`。物块初始 XY `(0.22,0.48)`、目标 XY `(-0.08,0.76)`，HOME XY `(0,0.48)`，工具初始偏航旋转 90°。可修改：

- [桌面与 HOME 配置](simulation/configs/calibrated_demo_y_table.json)
- [初始与目标点](simulation/configs/calibrated_demo_y_case.json)
- [仿真机器人配置](simulation/configs/xhand_local.json)

冻结策略为 `models/policies/xhand_pick_place_ppo.pt`，稳定快照为 `models/snapshots/xhand_stable_grasp.json`，不依赖历史 runs/ 中的副本。相机位姿沿用真机标定结果，不随桌面布局重新定向。

## 可下载的演示结果

GitHub 版本包含 [demos/calibrated_pick_place/](demos/calibrated_pick_place/README.md)：已验证的腕部/全局视频、首尾截图、轨迹、帧位姿、判定结果与当次源码快照。下载源码后即可查看这一演示；重新仿真仍需安装外部运行环境。

![已标定全局相机：完成抓放](demos/calibrated_pick_place/global_last.png)

历史 `simulation/runs/` 继续忽略；仅将这个指定案例整理为演示样例。公开内容范围和上传命令见 [docs/GITHUB_UPLOAD.md](docs/GITHUB_UPLOAD.md)。

## 真机标定：独立环境

在连接 D405、D435 和 UR5e 的 Ubuntu 电脑上，从合并仓库根目录创建 Python 3.10–3.12 环境：

```bash
python3 -m venv .venv-calibration
source .venv-calibration/bin/activate
unset PIP_CONSTRAINT
python -m pip install --upgrade pip
python -m pip install -e './calibration[hardware]'

bash scripts/calibrate.sh --help
bash scripts/calibrate.sh devices
```

如果只做离线求解，安装 `python -m pip install -e ./calibration` 即可。硬件 SDK 仅在设备采集命令中导入。已有独立环境可设置 `CALIBRATION_PYTHON=/path/to/python`。

`scripts/calibrate.sh` 自动进入 calibration/，所以相对参数路径都以该目录为基准：

```bash
bash scripts/calibrate.sh robot-state --config configs/local.json
bash scripts/calibrate.sh validate \
  --session data/validation_wrist \
  --intrinsics results/wrist_intrinsics_v3.json \
  --result results/handeye.json \
  --out results/validation_wrist_new.json
```

采集只读取机器人状态，不发送运动命令。完整原始流程见 [calibration/README.md](calibration/README.md)；相机序列号、局域网 IP 等机器配置在 `calibration/configs/local.json`，本地已保留，Git 忽略。

## 最终标定与检查

当前 Demo 使用的参数全部位于本仓库：

- `calibration/results/wrist_intrinsics_v3.json`、`global_intrinsics_v3.json`：最终内参。
- `calibration/results/handeye.json`、`global_extrinsics.json`：真机手眼与全局外参。
- `calibration/configs/link_to_tcp.confirmed.json`：已确认 wrist_3_link → TCP。
- `calibration/results/camera_rig_isaac.json`、`world_base_isaac.json`：仿真正式入口。
- `calibration/examples/isaac_camera_rig.py`：已验证 rectified 相机适配器，使用 `rig.rgb()` 取图。

坐标为米、列向量；真实 TCP 偏置为零，`T_link_tcp=I`；当前 env_0 的 `T_world_base` 为 Rz(pi) 加 `[0,0,0.72]` 平移。原确认包保持原样存档在 `archives/calibration_confirmed_20261023/`；报告内绝对路径与哈希是当时的溯源信息。

静态输入/模型/归档检查仅使用 Python 标准库：

```bash
python3 scripts/check_repo.py
```

在独立标定环境中，重算已复制的真实验证数据：

```bash
.venv-calibration/bin/python scripts/verify_calibration.py \
  --out "verification/real_validation_$(date -u +%Y%m%dT%H%M%S)"
```

现有标定测试：

```bash
PYTHONPATH=calibration/src .venv-calibration/bin/python \
  -m unittest discover -s calibration/tests -v
```

双相机联合交叉验证现阶段暂缓，不作为继续仿真视觉采集的前置条件。已完成的是各路固定板独立验证、真实/仿真 TCP 对比以及仿真投影检查。

## 新终端、迁移和备份

只配置仿真终端时使用 `source env/setup_sim.sh`；该命令会进入 simulation/。新机器可先复制 `env/local.env.example` 为 `env/local.env`，配置本机 Python 和可选 NVIDIA 用户态库路径。脚本选择已存在的环境/库，不安装或更换服务器内核驱动。

项目源码、资产和模型使用相对位置，可整体移动。Isaac Sim 5.0.0 / Isaac Lab 2.2.1 / Python 3.11 / PyTorch 2.7.0+cu128、系统图形库与匹配的 NVIDIA 驱动需在目标机器另行配置。

当前服务器的 Isaac Lab 是从原 `tweezers_xhand_w01/software/IsaacLab-2.2.1` 编辑安装到外部仿真环境的，原软件目录继续作为 SDK 依赖使用。若后续清理原目录，应先迁移或重新安装这个外部 SDK；合并没有修改已运行成功的 Python 安装环境。

Git 默认保存代码、资产、冻结模型及标定 JSON；**Git 克隆不会包含已忽略的原始采集数据、历史视频和验证归档**。完整备份在仓库外执行：

```bash
tar -C /mnt/workspace -czf /mnt/workspace/tweezers-sim2real-full.tar.gz tweezers-sim2real
```

CPFS 配额恢复后，可整体复制回挂载存储：

```bash
cp -a /mnt/workspace/tweezers-sim2real /cpfs/user/xiaoqihuang/
cd /cpfs/user/xiaoqihuang/tweezers-sim2real
python3 scripts/check_repo.py
bash scripts/run_demo.sh
```

历史 simulation/README.md 及其他技术报告保留原阶段内容，部分引用原作者机器路径或不随包提供的旧实验。本 README 与统一 scripts/ 是合并后的当前入口。来源与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，进度见 [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md)，整理范围见 [docs/MERGE.md](docs/MERGE.md)。
