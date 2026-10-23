# 使用教程：选择 XHand 或 Sharpa W01，用镊子夹取方块

更新日期：2026-10-21。面向第一次接触本项目、需要在当前服务器或另一台机器上运行的使用者。

本项目在 Isaac Sim 中模拟 **UR5e 机械臂 + 灵巧手 + 被动弹簧镊子**。机器人从已经稳定握住镊子的初态开始，接近方块、夹紧、抬起、搬运、放下、松开并撤离。方块和镊子由物理接触驱动。

本教程不包含从桌面捡起镊子、相机识别方块或连接真实机器人。当前任务直接读取仿真状态。两套手都能运行，但需要各自的模型、握姿和初始快照。

## 1. 先选择要运行的版本

| 选择 | XHand v15 | Sharpa Wave W01 v6d |
|---|---|---|
| 机械臂 | UR5e | UR5e |
| 控制方式 | 几何阶段控制器 + 已训练的残差 PPO | 几何阶段控制器，尚无 W01 专用 PPO |
| 入口 | `scripts/run_pick_place.py` | `scripts/run_sharpa_w01.sh`，内部调用同一个 Python 运行器 |
| 策略参数 | `--evaluate <XHand检查点.pt>` | `--evaluate zero`，包装脚本已默认设置 |
| 触觉记录 | 基础接触与夹持指标 | 可选五指 TacMap 几何触觉和独立 PhysX 接触力 |
| 项目定位 | 保留的历史基线 | 当前主要开发版本 |

这里没有统一的 `--hand xhand/sharpa` 参数。选择手型实际是选择 **运行入口、`--pose` 配置、`--snapshot` 快照和相应控制参数**。不能把 XHand 的 PPO 权重直接用在 W01 上。

建议顺序：准备文件 → 安装或选择环境 → 生成本机路径配置 → 运行一个固定案例 → 验证结果 → 按需录像、记录触觉或批量评估。

## 2. 获取完整项目及模型资产

### 2.1 只有代码仓库不够

请向项目提供者取得当前项目文件和下面的资产。项目 `.gitignore` 忽略了 `runs/`、`outputs/`，所以仅 `git clone` 未必包含已有策略和初始快照。不要把整个 `runs/` 当作缓存删除。

本教程采用以下目录布局；`work/` 可以放在任意有读写权限的位置：

```text
work/
├── assets/
│   └── ur5e_xhand_right/          # 外部资产，必须由提供者一并交付
│       ├── robot.urdf
│       ├── robot.usd
│       ├── configuration/        # USD 引用的各子层
│       ├── sources/              # UR5e 和 XHand 网格、来源及许可证
│       └── ...
└── tweezers/                     # 本项目根目录，下文称 PROJECT
    ├── src/
    ├── scripts/
    ├── analysis/
    ├── configs/
    ├── assets/
    ├── runs/
    └── TUTORIAL.md
```

**外部 `work/assets/ur5e_xhand_right/` 对两套手都很重要。** XHand 使用其中的整套机器人；当前 W01 组合 URDF 也通过相对路径引用其中的 UR5e 网格。因此，即使只运行 W01，也要提供该目录。请完整复制目录，包括 `sources/`、`configuration/` 和许可证，不要只复制 `robot.usd`。

原服务器的两个位置为：

```text
项目：/cpfs/user/lvzhaoyang/work/ai_rl/tweezers
外部资产：/cpfs/user/lvzhaoyang/work/ai_rl/assets/ur5e_xhand_right
```

保持上面的相对布局，可以直接复用 W01 URDF 中现有的网格引用，无需重新导入或重建机器人。

### 2.2 必须保留的文件

除完整的 `src/`、`scripts/`、`analysis/`、`configs/` 外，至少核对以下内容：

| 用途 | 相对项目根目录的位置 |
|---|---|
| XHand 握姿 | `configs/human_grasp_middle_backup_v14.json` |
| XHand 稳定预持镊快照 | `runs/middle_backup13_level_search/m9_c0_snapshot.json` |
| XHand 已训练策略 | `runs/pick_place15_ppo101_retreat80/iteration_00025.pt` |
| XHand 固定示例及相机 | `configs/pick_place_preview_case.json`、`analysis/pick_place_cameras.json` |
| W01 组合机器人 | 完整的 `assets/ur5e_sharpa_w01_right/` |
| W01 官方原始模型、网格及许可证 | 完整的 `assets/sharpa_wave/official/` |
| W01 握姿及地图选择 | `configs/sharpa_w01_natural_v6d_tacmap_matched_v2.json` |
| W01 稳定预持镊快照 | `configs/sharpa_w01_natural_v6c_preheld.json` |
| W01 固定示例及相机 | `configs/sharpa_w01_hard_case.json`、`configs/sharpa_w01_tacmap_cameras.json` |
| W01 官方 TacMap 代码及原始表面 | 完整的 `assets/sharpa_tactile/` |
| W01 匹配表面 | 完整的 `assets/sharpa_tacmap_w01_matched_v2/` |
| TacMap 兼容代码及许可证 | 完整的 `src/sharpa_tacmap_vendor/` |
| 视频合成中文字体 | `assets/fonts/NotoSansCJKsc-Regular.otf` |

官方 W01 单手模型和本项目的 UR5e+W01 组合模型不是同一个资产。首次使用请取得已构建的组合资产；重建说明见 [README_SHARPA_W01.md](README_SHARPA_W01.md)。

## 3. 准备运行环境

### 3.1 环境选择和硬件

本项目已有运行记录使用 **Isaac Sim 5.0.0 + Isaac Lab 源码标签 v2.2.1 + Python 3.11**，GPU 为 RTX 4090。下面的新机器安装路线也固定这组版本，避免直接安装最新版造成接口变化。

新机器建议使用 Linux x86_64、Ubuntu 22.04/24.04 和具备 RTX 渲染能力的 NVIDIA GPU。先运行 `nvidia-smi` 确认驱动和 GPU 可用。显存需求随环境数、相机和触觉记录增加，第一次使用 `--num-envs 1`；本教程未测定最低显存。仅 CPU 环境不能复现这里的 CUDA 物理与相机流程。

新机器安装参考 [Isaac Lab v2.2.1 官方安装文档](https://isaac-sim.github.io/IsaacLab/v2.2.1/source/setup/installation/pip_installation.html)。下述命令是按官方安装流程和项目依赖整理的迁移方案；本次编写教程没有在空白机器上重新安装整套环境。

### 3.2 路线 A：使用当前服务器已有环境

在当前服务器上无需重装 Isaac Sim/Lab。在 Bash 中执行：

```bash
cd /cpfs/user/lvzhaoyang/work/ai_rl/tweezers
export PROJECT="$PWD"
export ISAAC_PYTHON=/cpfs/user/lvzhaoyang/apps/bin/isaac-python
"$ISAAC_PYTHON" --version
nvidia-smi
```

`isaac-python` 是本机包装脚本，负责加载 Isaac Sim Python 和图形运行库。不要把这条绝对路径复制到另一台机器上后直接使用。后续 `ISAAC_PYTHON` 必须指向**一个可执行文件**，不能写成带参数的字符串，例如 `python -u`。

当前服务器若需给仿真环境补包，应沿用 `/cpfs/user/lvzhaoyang/apps/ISAAC_LAB.md` 的约束方式，避免系统容器的全局 pip 约束覆盖仿真器 PyTorch；不要在已有环境直接执行下一节的全新安装命令。

### 3.3 路线 B：在新机器安装环境

准备好 Git、Python 3.11 及其 `venv` 模块。Ubuntu 的默认 Python 不一定是 3.11，请先确认 `python3.11 --version`。使用已有 Conda 的用户也可先创建 Python 3.11 环境；下面以 `venv` 为例。

先把项目放到第 2 节的布局中，将第一行替换为实际路径：

```bash
cd /你的路径/work/tweezers
export PROJECT="$PWD"
python3.11 -m venv "$PROJECT/.venv-sim"
source "$PROJECT/.venv-sim/bin/activate"
export ISAAC_PYTHON="$PROJECT/.venv-sim/bin/python"
python --version
nvidia-smi
```

安装仿真环境。以下约束文件仅用于这个新环境；不要把系统 Python 的另一套 Torch 约束带进来：

```bash
cat > "$PROJECT/.venv-sim/project-constraints.txt" <<'EOF'
torch==2.7.0+cu128
torchvision==0.22.0+cu128
numpy<2
warp-lang==1.7.1
setuptools<81
transformers<5
pin==2.7.0
cmeel-boost==1.83.0
eigenpy==3.5.1
hpp-fcl==2.4.4
EOF

export PIP_CONSTRAINT="$PROJECT/.venv-sim/project-constraints.txt"
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'isaacsim[all,extscache]==5.0.0' \
  --extra-index-url https://pypi.nvidia.com

mkdir -p "$PROJECT/../software"
git clone --branch v2.2.1 --depth 1 \
  https://github.com/isaac-sim/IsaacLab.git "$PROJECT/../software/IsaacLab-2.2.1"
cd "$PROJECT/../software/IsaacLab-2.2.1"
./isaaclab.sh --install none
cd "$PROJECT"

python -m pip install scipy pillow imageio imageio-ffmpeg
unset PIP_CONSTRAINT
```

`--install none` 安装 Lab 扩展，跳过可选第三方强化学习框架；本项目 PPO 实现在 `src/ppo.py`，运行它不需要额外安装 Stable-Baselines3、RL-Games 或 RSL-RL。若 `software/IsaacLab-2.2.1` 已存在，应确认是正确版本再复用，不必重复 clone。

上述 NumPy、Warp 等约束来自当前服务器的安装记录；它们不是所有传递依赖的完整锁文件。安装完成后可用 `python -m pip freeze > .venv-sim/installed-packages.txt` 留存本机版本。

普通包优先阿里云 HTTPS 镜像；若缺包，按需要将对应命令的索引换成 `https://pypi.org/simple/`。PyTorch CUDA 包和 NVIDIA 仿真器仍需访问各自的下载源。当前服务器普通联网先直连，连接失败才用已提供的 `with-general-proxy <命令及参数>` 重试；该包装命令不是第三方机器的通用工具。不要自动改用 PJLab，也不要把原服务器代理凭据复制到新机器。

Isaac Sim 首次启动可能需要下载扩展和编译缓存。按首次提示阅读并接受 NVIDIA 软件许可后继续。无界面自动运行时，可在接受许可后设置 `OMNI_KIT_ACCEPT_EULA=YES`；只有确实以 root 运行时才按 Kit 提示设置 `OMNI_KIT_ALLOW_ROOT=1`。官方首次启动说明见 [安装及验证步骤](https://isaac-sim.github.io/IsaacLab/v2.2.1/source/setup/installation/pip_installation.html#verifying-the-isaac-sim-installation)。

先验证基础环境：

```bash
"$ISAAC_PYTHON" -c 'import torch; from isaaclab.app import AppLauncher; print("CUDA:", torch.cuda.is_available()); print("Torch:", torch.__version__)'

# 持续运行的官方空场景示例；看到场景启动后按 Ctrl+C 结束。
"$ISAAC_PYTHON" "$PROJECT/../software/IsaacLab-2.2.1/scripts/tutorials/00_sim/create_empty.py" --headless
```

只有导入成功不代表物理和渲染已正常；后续固定案例才是任务级检查。项目内的 TacMap 兼容代码会在需要时自动加载，不要为了 TacMap 把整个 Lab 升级成 2.3.2。

### 3.4 单独准备结果分析和视频合成环境

这一步两条路线都适用。它与仿真环境隔离，避免分析工具升级 NumPy 或 USD 后影响 Isaac Sim。

```bash
cd "$PROJECT"
python3 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  numpy scipy pillow imageio imageio-ffmpeg usd-core
.venv-analysis/bin/python -c 'from pxr import Usd, UsdPhysics; import numpy, scipy, imageio; print("analysis imports OK")'
```

`usd-core` 在此仅供离线读取 USD；不要把它装进仿真环境去替换 Isaac Sim 自带的 USD。`imageio-ffmpeg` 提供视频编码后端，中文合成还需第 2 节列出的字体。本教程的基础验收不需要 MediaPipe、Blender、ROS 或额外的 `python-fcl`。

## 4. 生成本机手型配置并检查路径

原始配置带有开发服务器路径：W01 的 `robot_asset_dir` 是绝对路径，XHand 未显式设置该字段时会回退到 `src/kinematics.py` 中的本机路径。下面生成两份本机配置，保留原始配置和实验记录不变。

**当前服务器和新机器均执行本节**，这样后面的命令一致。必须先完成第 2 节的目录布局。脚本还显式保留 XHand 原先的两组机构碰撞过滤，避免仅改变模型路径就改变物理行为。

```bash
cd "$PROJECT"
python3 - <<'PY'
import json
from pathlib import Path

root = Path.cwd().resolve()
xhand_asset = root.parent / "assets/ur5e_xhand_right"
w01_asset = root / "assets/ur5e_sharpa_w01_right"
for asset in (xhand_asset, w01_asset):
    for name in ("robot.urdf", "robot.usd"):
        assert (asset / name).is_file(), f"缺少模型文件：{asset / name}"

xhand = json.loads((root / "configs/human_grasp_middle_backup_v14.json").read_text())
xhand["robot_asset_dir"] = str(xhand_asset)
xhand["mechanical_self_collision_pairs"] = [
    ["wrist_3_link", "right_hand_index_rota_link1"],
    ["wrist_3_link", "right_hand_thumb_rota_link1"],
]
w01 = json.loads((root / "configs/sharpa_w01_natural_v6d_tacmap_matched_v2.json").read_text())
w01["robot_asset_dir"] = str(w01_asset)
w01["tacmap_surface_directory"] = "assets/sharpa_tacmap_w01_matched_v2"

for name, config in [("xhand_local.json", xhand), ("sharpa_w01_local.json", w01)]:
    target = root / "configs" / name
    with target.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print("已生成", target)
PY
```

文件已存在时脚本会停止，避免覆盖已有本机设置；确认原文件路径正确即可继续。以后移动项目或外部资产时，应更新这两份配置的 `robot_asset_dir`。

再检查主要运行文件及 URDF 的全部网格引用：

```bash
cd "$PROJECT"
python3 - <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, "src")
from kinematics import Kinematics

required = [
    "runs/middle_backup13_level_search/m9_c0_snapshot.json",
    "runs/pick_place15_ppo101_retreat80/iteration_00025.pt",
    "configs/sharpa_w01_natural_v6c_preheld.json",
    "configs/pick_place_preview_case.json",
    "configs/sharpa_w01_hard_case.json",
]
for name in required:
    assert Path(name).is_file(), f"缺少复现文件：{name}"
for name in ("xhand_local.json", "sharpa_w01_local.json"):
    config = json.loads((Path("configs") / name).read_text())
    asset = Path(config["robot_asset_dir"])
    kin = Kinematics(asset / "robot.urdf",
        palm_link=config.get("palm_link", "right_hand_link"),
        hand_root_link=config.get("hand_root_link"),
        arm_joint_names=config.get("arm_joint_names"))
    missing = [str(kin.resolve_mesh(e.get("filename")))
        for e in kin.root.findall(".//mesh")
        if not kin.resolve_mesh(e.get("filename")).is_file()]
    assert not missing, f"{name} 缺少网格：{missing}"
    assert (asset / "configuration").is_dir(), f"缺少 USD 子层目录：{asset}"
    print(name, "检查通过，主动关节数：", len(kin.active))
PY
```

这段检查导入了 NumPy 和 SciPy；若系统 `python3` 没有它们，将第二段命令的 `python3` 换成 `.venv-analysis/bin/python`。检查通过只表示这些文件齐全，仍需实际运行确认 USD 加载和物理仿真。

## 5. 选择 XHand：运行已训练策略

所有任务命令都在项目根目录执行。`--out` 必须是不存在的新目录；第二次运行请换一个名字。

### 5.1 一个固定案例，不录视频

```bash
cd "$PROJECT"
"$ISAAC_PYTHON" scripts/run_pick_place.py \
  --out runs/tutorial_xhand_case01 \
  --pose configs/xhand_local.json \
  --snapshot runs/middle_backup13_level_search/m9_c0_snapshot.json \
  --evaluate runs/pick_place15_ppo101_retreat80/iteration_00025.pt \
  --cases configs/pick_place_preview_case.json \
  --num-envs 1 --seed 0 --retreat-closure .8 --device cuda:0
```

`--retreat-closure .8` 是选定 XHand v15 策略的复现参数，不能省略后按入口默认的 `.7` 运行。`--evaluate` 指定已有策略，因而不需要重新训练。

### 5.2 运行并录制视频

```bash
"$ISAAC_PYTHON" scripts/run_pick_place.py \
  --out runs/tutorial_xhand_video01 \
  --pose configs/xhand_local.json \
  --snapshot runs/middle_backup13_level_search/m9_c0_snapshot.json \
  --evaluate runs/pick_place15_ppo101_retreat80/iteration_00025.pt \
  --cases configs/pick_place_preview_case.json \
  --camera-config analysis/pick_place_cameras.json \
  --num-envs 1 --seed 0 --retreat-closure .8 --device cuda:0
```

运行器在代码中强制无界面运行；不弹窗口是正常现象。提供 `--camera-config` 后会启用相机，输出 `overview.mp4` 和 `task.mp4`。这不是需要 ROS 控制的交互程序。

### 5.3 批量评估或重新训练（可选）

不传 `--cases` 时，起点和目标在配置工作区内随机采样。以下重现历史评估的设置，但新机器结果仍须独立验收：

```bash
"$ISAAC_PYTHON" scripts/run_pick_place.py \
  --out runs/tutorial_xhand_eval64 \
  --pose configs/xhand_local.json \
  --snapshot runs/middle_backup13_level_search/m9_c0_snapshot.json \
  --evaluate runs/pick_place15_ppo101_retreat80/iteration_00025.pt \
  --num-envs 64 --seed 20261017 --retreat-closure .8 --device cuda:0
```

若要从随机网络训练，省略 `--evaluate` 并明确设置训练参数：

```bash
"$ISAAC_PYTHON" scripts/run_pick_place.py \
  --out runs/tutorial_xhand_train \
  --pose configs/xhand_local.json \
  --snapshot runs/middle_backup13_level_search/m9_c0_snapshot.json \
  --num-envs 64 --seed 101 --iterations 25 \
  --learning-rate 1e-5 --std .04 --retreat-closure .8 --device cuda:0
```

新训练不自动继承旧模型成绩。`--evaluate zero` 可用于 XHand 名义控制器对照，但它不是已训练 PPO 的等价替代。没有检查点时应取得检查点或重新训练，不要把缺失权重当作环境安装问题。

## 6. 选择 Sharpa W01：运行阶段控制器

### 6.1 一个固定案例，不录视频

```bash
cd "$PROJECT"
bash scripts/run_sharpa_w01.sh \
  --out runs/tutorial_w01_case01 \
  --pose configs/sharpa_w01_local.json \
  --snapshot configs/sharpa_w01_natural_v6c_preheld.json \
  --cases configs/sharpa_w01_hard_case.json \
  --num-envs 1 --evaluate zero --device cuda:0
```

包装脚本会读取前面导出的 `ISAAC_PYTHON`，并提供 W01 对应的关节驱动、闭合量、速度和释放参数。第三方机器必须设置它，否则会回退到原服务器的解释器路径。

不要只在 XHand 命令中换一个模型路径：W01 还需要配套的快照和控制参数。教程使用的 v6d 握姿与 v6c 快照是当前入口已有的配对，不是版本误写。

### 6.2 录制仿真、五指触觉和接触力

```bash
bash scripts/run_sharpa_w01.sh \
  --out runs/tutorial_w01_video01 \
  --pose configs/sharpa_w01_local.json \
  --snapshot configs/sharpa_w01_natural_v6c_preheld.json \
  --cases configs/sharpa_w01_hard_case.json \
  --camera-config configs/sharpa_w01_tacmap_cameras.json \
  --num-envs 1 --evaluate zero --device cuda:0 \
  --tactile --tacmap
```

若只想录制手与方块的运动，去掉最后一行的 `--tactile --tacmap` 即可。

- `--tactile`：保存独立 PhysX 接触力、接触位置及相关映射数据。
- `--tacmap`：调用官方 TacMap 算法，保存五指几何压入灰度图；它不是压力图或经过实物标定的皮肤形变。
- 触觉记录入口要求 `--num-envs 1`。普通、不带触觉的任务可使用多个环境。

本教程首先使用 `run_sharpa_w01.sh` 的刚性接触基线。另有 `run_sharpa_w01_compliant.sh` 柔顺实验入口，但参数、快照不同，尚未达到刚性基线的回归表现。首次使用不混用两条路线；柔顺实验详见 [README_SHARPA_W01_NATURAL.md](README_SHARPA_W01_NATURAL.md)。

## 7. 检查结果并导出视频

### 7.1 任务是否真正成功

成功不能只依据程序退出或视频中“看起来抬起了”。评估模式会生成：

| 文件 | 用途 |
|---|---|
| `evaluation.json` | 逐环境成功结果和任务配置 |
| `trajectory.npz` | 方块、镊子、关节、阶段及接触等轨迹 |
| `launch.json`、`source/` | 启动参数、相关配置和源码快照 |
| `final_scene.usda` | 导出的场景结构，不等同于完整运动录像 |
| `overview.mp4`、`task.mp4` | 启用相机后生成的原生视频 |
| `tactile.npz`、`tacmap.npz` | 启用对应 W01 记录后生成的数据 |

例如检查第 5.1 节的 XHand 结果：

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("runs/tutorial_xhand_case01/evaluation.json")
r = json.loads(p.read_text())
print("各环境是否成功：", r["success_per_env"])
PY

.venv-analysis/bin/python analysis/validate_pick_place.py \
  runs/tutorial_xhand_case01 --geometry
```

W01 使用相同验证器，将运行目录换成 `runs/tutorial_w01_case01`。录像案例也分别验证对应的 `*_video01` 目录。

验证器生成 `validation.json`。检查 `all_claims_consistent`，并确认 `reconstructed_successes` 等于预期成功案例数；失败结果也可能被一致地重构，因此“一致”不等于“全部成功”。基础判据包括双侧夹持并抬升、目标处稳定放置、松开撤离以及镊子仍被保持。`--geometry` 另查控制采样点的源网格桌面间隙和关节限位，不代表连续时间所有自碰撞均已证明不存在。

### 7.2 合成 XHand 视频

先对录像对应目录运行任务验证，再执行：

```bash
.venv-analysis/bin/python analysis/validate_pick_place.py \
  runs/tutorial_xhand_video01 --geometry
mkdir -p outputs
.venv-analysis/bin/python scripts/compose_pick_place_video.py \
  runs/tutorial_xhand_video01 --output outputs/tutorial_xhand_video01.mp4
```

这个合成器用于 XHand，且会拒绝把失败回合合成为成功展示。不要用它合成 W01，否则标题和策略说明不适用。已有视频输出也应使用新名称保留。

### 7.3 验证 W01 触觉并合成视频（可选）

TacMap 验证需要独立固定几何对照。**在新机器上重新生成对照**，因为历史对照包含原机器的源码路径和哈希，不能只复制旧结果就认为新环境验证通过。

```bash
"$ISAAC_PYTHON" analysis/sharpa_tacmap_controls.py \
  --out runs/tutorial_w01_controls01 \
  --surface-directory assets/sharpa_tacmap_w01_matched_v2 \
  --resolution-step 1 --headless --device cuda:0

.venv-analysis/bin/python analysis/validate_pick_place.py \
  runs/tutorial_w01_video01 --geometry
.venv-analysis/bin/python analysis/validate_sharpa_tactile.py \
  runs/tutorial_w01_video01 --require-video --require-task-success
.venv-analysis/bin/python analysis/validate_sharpa_tacmap.py \
  runs/tutorial_w01_video01 --require-video --require-task-success \
  --controls runs/tutorial_w01_controls01

mkdir -p outputs
.venv-analysis/bin/python scripts/compose_sharpa_tacmap_video.py \
  runs/tutorial_w01_video01 --output outputs/tutorial_w01_video01.mp4 \
  --require-validated --fps 30 --max-mb 18 \
  --contact-note '刚性接触基线；TacMap 灰度为几何响应，非压力'
```

对照目录也使用新名称。固定几何对照验证传感器算法，不验证真实材料。项目已发现官方多目标 CPD 在双镊臂场景中的几何歧义；即便标准验证通过，也不能据此认定所有像素都对应正确的物理接触。相关诊断和逐目标复算见 [W01 当前说明](README_SHARPA_W01_NATURAL.md)。

## 8. 指定自己的方块起点和目标

创建新的 JSON 文件，例如 `configs/tutorial_cases.json`：

```json
{
  "start": [[0.55, -0.18]],
  "goal": [[0.72, 0.05]]
}
```

然后把第 5 或第 6 节命令中的参数换成：

```bash
--cases configs/tutorial_cases.json --num-envs 1
```

坐标为任务基座桌面坐标系下的 XY，单位米；不需要填写方块高度。多个案例时，`start` 和 `goal` 必须分别有 N 行，并设置 `--num-envs N`。W01 触觉仍只支持 N=1。

默认采样区域为 X∈[0.48, 0.76] m、Y∈[−0.22, 0.08] m，随机起终点间距至少 0.12 m。显式 `--cases` 不会自动将输入限制到可靠范围，建议遵守该区域和间距。W01 在此区域并非所有案例都已成功，修改坐标后应重新检查结果。

## 9. 常见问题

| 现象 | 检查与处理 |
|---|---|
| `ModuleNotFoundError: isaaclab` / `isaacsim` | 确认使用 `"$ISAAC_PYTHON"`，且 Lab 安装在同一解释器中；普通系统 Python 和分析环境不负责启动仿真 |
| 找不到 `/cpfs/...` | 未设置新机器 `ISAAC_PYTHON`，或未传第 4 节生成的本机 `--pose` |
| `robot.urdf`、网格或 USD 子层缺失 | 按第 2 节完整复制外部资产及项目资产，保持相对布局；不要只复制主 USD |
| XHand 权重或初态快照缺失 | 这些在被 Git 忽略的 `runs/` 中，需由项目提供者另行交付 |
| 本机配置已存在 | 复查其中的路径后复用；生成步骤使用独占创建，不覆盖现有设置 |
| `FileExistsError` 或输出目录拒绝覆盖 | 将 `--out` 改成新的目录，不覆盖历史实验 |
| 没有弹出仿真窗口 | 当前任务强制 headless，使用 `--camera-config` 获取录像 |
| `pxr` 导入失败，验证器寻找原服务器路径 | 用第 3.4 节的分析环境并确认 `usd-core` 可导入；未安装时旧验证器会尝试本机 Isaac Sim 回退路径 |
| 缺少 `multi_mesh_ray_caster` 等 TacMap 接口 | 检查 `src/sharpa_tacmap_vendor/` 完整性，使用项目入口自动加载兼容层 |
| CUDA/RTX/Vulkan 启动失败 | 先排查 NVIDIA 驱动和容器 GPU/图形能力，再运行官方空场景；Torch 能看到 GPU 不等于 RTX 渲染已可用 |
| 显存不足 | 从单环境、无相机、无触觉开始，按需增加功能；用 `--device cuda:N` 选卡，避免再混用 `CUDA_VISIBLE_DEVICES` 改变编号 |
| 视频编码或字体失败 | 检查分析/仿真环境各自的 `imageio-ffmpeg`，以及 `assets/fonts/NotoSansCJKsc-Regular.otf` |
| 程序正常结束但夹取失败 | 查看 `success_per_env` 和独立验证；不要仅通过放宽成功判据掩盖失败 |

每次打开新终端，需要重新设置 `PROJECT`、`ISAAC_PYTHON`；新机器还需激活相应虚拟环境。运行中不要修改源码、手型配置或资产，执行来源检查可能将其标为漂移。

## 10. 结果范围和进一步阅读

已有历史记录中，XHand v15 独立测试为 64/64；W01 v6d 刚性基线旧坐标回归为 8/8，新随机测试为 6/8；W01 柔顺实验同八组回归为 5/8。这些是不同测试集和配置下的历史结果，不是可直接横向比较的统一基准，也不保证新机器或新坐标得到相同成绩。

- [README.md](README.md)：当前工作状态与问题。
- [README_XHAND.md](README_XHAND.md)：XHand 历史策略、批量测试和训练说明。
- [README_SHARPA_W01.md](README_SHARPA_W01.md)：W01 资产、许可证和机器人构建。
- [README_SHARPA_W01_NATURAL.md](README_SHARPA_W01_NATURAL.md)：W01 握姿、TacMap、柔顺实验和限制。

本文依据当前代码、配置、资产引用和安装记录编写，并核对了官方安装文档。路径和命令检查不替代新机器安装验收；迁移后请以自己保存的轨迹、验证报告和视频判断运行是否成功。
