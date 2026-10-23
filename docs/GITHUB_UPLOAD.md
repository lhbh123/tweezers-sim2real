# GitHub 公开版本

仓库根目录：`/mnt/workspace/tweezers-sim2real`。以下命令由用户执行；整理过程没有创建远程仓库或推送。

## 纳入 Git 的内容

- `calibration/`：标定代码、测试、示例、配置模板、确认 TCP 参数和顶层标定结果 JSON。原始采集数据与机器本地配置忽略。
- `simulation/`：仿真代码、配置、脚本、已有分析工具/报告以及内置资产，保留历史功能；忽略 runs 和缓存。
- `assets/`：完整 UR5e + XHand URDF、USD、网格及来源许可。
- `models/`：Demo 使用的 PPO checkpoint 与稳定抓握快照。
- `demos/calibrated_pick_place/`：成功案例的两路 MP4、首尾 PNG、轨迹、相机帧位姿、判定结果及当次源码/输入快照。不上传约 90.7 MiB 的展开场景 final_scene.usda。
- `scripts/`、`env/`、`docs/`、根文档与来源记录：统一入口、环境模板、验证摘要、参考照片与实际进度。

虚拟环境、Isaac Sim/Isaac Lab/CUDA/驱动安装、下载器、本地 env/local.env 和 calibration/configs/local.json、原始标定数据、历史 runs、verification、archives 及备份压缩包不进入 Git。保留第三方许可，不把第三方资产或权重重新声明为本项目统一开源许可。

## 首次上传

前提：已安装 gh 并登录；目标 GitHub 仓库名尚未使用。该流程将创建一个新的公开仓库，不会覆盖旧标定仓库。

```bash
(
  set -e
  cd /mnt/workspace/tweezers-sim2real

  gh auth status
  gh auth setup-git
  GH_USER="$(gh api user --jq .login)"
  GH_ID="$(gh api user --jq .id)"

  # 已有 Git 身份则沿用；否则设置本仓库的身份与 GitHub 隐私邮箱。
  if ! git config user.name >/dev/null; then
    git config --local user.name "$GH_USER"
  fi
  if ! git config user.email >/dev/null; then
    git config --local user.email "${GH_ID}+${GH_USER}@users.noreply.github.com"
  fi

  python3 scripts/check_repo.py
  git add .
  git diff --cached --stat
  if ! git diff --cached --quiet; then
    git commit -m "Initial release: calibrated UR5e XHand tweezer pick and place"
  fi
  git branch -M main

  gh repo create "$GH_USER/tweezers-sim2real" \
    --public \
    --source=. \
    --remote=origin \
    --description "UR5e + XHand tweezer pick and place: RealSense calibration, Isaac Sim, assets and dual-view demo" \
    --push

  gh repo view --json url --jq .url
)
```

当前文件均低于 GitHub 普通 Git 的 100 MiB 单文件限制，此版本无需 Git LFS。整体包含机器人网格/模型和视频，首次推送需要上传二进制文件。

这份 GitHub 版本可复现当前指定 Demo 的代码和资产输入；原始标定数据未上传，因此原始真机数据复算命令需要自行恢复采集数据。静态检查和合成标定测试可以使用 Git 克隆版本运行。新机器运行仿真还需另行安装 README 指定的软件与驱动。
