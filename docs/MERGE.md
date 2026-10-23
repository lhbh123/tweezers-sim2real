# 2026-10-23 仓库合并记录

原目录 `/cpfs/user/xiaoqihuang/xhand-camera-calibration` 和 `/cpfs/user/xiaoqihuang/tweezers_xhand_w01` 保留。合并工作副本位于 `/mnt/workspace/tweezers-sim2real`，原因是 CPFS 新建目录返回 `Errno 122: Disk quota exceeded`。当前路径在实例系统盘，迁移实例前需另存到持久存储。

| 原内容 | 合并后位置 |
|---|---|
| xhand-camera-calibration 源码、测试、配置 | calibration/ |
| 真机原始 data/ 与 results/ | calibration/data/、calibration/results/，数据不加入 Git |
| tweezers_xhand_w01/tweezers/ | simulation/，含历史 runs/ |
| tweezers_xhand_w01/assets/ | assets/ |
| 运行必需 PPO 权重与稳定快照 | models/policies/、models/snapshots/，加入 Git |
| 已确认相机适配器、正式 rig、world/base | calibration/examples/、calibration/results/ |
| 原确认目录、交接 ZIP、原清单 | archives/，保持原始证据，Git 忽略 |
| 原项目进度文档 | PROJECT_PROGRESS.md，路径已更新 |

合并统一了入口与路径，没有重新求解标定、重新训练策略或改变任务物理。XHand 资产路径已改为仓库内相对路径；适配器采用通过投影检查的 rectified 版本，替代原标定仓库中尚未修正的示例。采集数据和历史结果内的绝对路径保留作来源记录。

未复制 `.git`、虚拟环境、egg-info、字节码、software/ 和下载的软件安装器。原标定仓库 Git 历史另存到 archives/calibration-git-history.bundle；原提交、输入 SHA-256 以及工作区未提交文件记录于 MERGE_PROVENANCE.json。原仓库未提交的 docs/T_LINK_TCP_CONFIRMATION.md 也已复制。

仿真与标定继续使用独立环境。当前服务器仅复用已有仿真 Python 执行离线迁移验证，没有安装、升级或删除包，也未连接真实相机/机器人。

检查结果写入 verification/，新仿真结果写入 simulation/runs/。冻结确认目录不用于可写运行输出，避免改动其原始清单。
