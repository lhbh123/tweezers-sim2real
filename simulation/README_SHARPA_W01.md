# Sharpa Wave W01：资产、构建与接口

当前实验为 W01 v6d 自然握姿与官方 TacMap，进展和问题见 [主 README](README.md)，完整复现见 [当前实验说明](README_SHARPA_W01_NATURAL.md)。本页维护资产与代码接口，不把旧 v4/v5 结果当作当前成绩。更新日期：2026-10-20。

新增的 [柔顺接触实验入口](scripts/run_sharpa_w01_compliant.sh)及独立逐目标 CPD 复核见上述实验说明的首节：固定案例通过，同起终点回归 5/8（旧基线 8/8），尚不替换旧默认。官方多目标原图存在跨镊臂射线误接受，须结合另存的逐目标诊断解释。

用户最初指定 W02，后明确同意使用 W01。本项目使用官方 `wave_01` 右手的 22 个主动关节；没有通过缩放或改名将 W01 当作 W02。原资产查询记录见 [asset_audit.json](analysis/sharpa_research/asset_audit.json)。

## 来源与许可证

| 内容 | 固定来源 | 本地位置 |
|---|---|---|
| W01 右手 URDF、USD、MJCF 与网格 | [sharpa-urdf-usd-xml](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml)，commit `0d19cac602f46456b819e4b6a2c09a74982c9a3e` | [assets/sharpa_wave/official/](assets/sharpa_wave/official/) |
| 官方 TacMap 代码、配套模型和原始表面地图 | [sharpa-tacmap](https://github.com/sharpa-robotics/sharpa-tacmap)，commit `e4f8827407e15625368ef0b9cf5a72c44304e16b` | [assets/sharpa_tactile/](assets/sharpa_tactile/) |
| 当前 W01 CAD 派生地图 | 本项目离线几何匹配，未按轨迹或接触力拟合 | [assets/sharpa_tacmap_w01_matched_v2/](assets/sharpa_tacmap_w01_matched_v2/) |
| Isaac Lab 兼容代码 | 上游 v2.3.2 的固定源码子集 | [来源记录和许可证](src/sharpa_tacmap_vendor/isaaclab_232/) |
| UR5e | 本机既有资产，输入哈希写入构建记录 | `/cpfs/user/lvzhaoyang/work/ai_rl/assets/ur5e_xhand_right/ur5e.urdf` |

W01 和 TacMap 的 Apache 2.0 LICENSE / NOTICE 原样保留：[W01 LICENSE](assets/sharpa_wave/official/LICENSE.txt)、[W01 NOTICE](assets/sharpa_wave/official/NOTICE.txt)、[TacMap LICENSE](assets/sharpa_tactile/LICENSE)、[TacMap NOTICE](assets/sharpa_tactile/NOTICE)。Isaac Lab 子集遵循随附 BSD 3-Clause；UR5e 和外部网格沿用各自来源许可。

W01 下载清单记录 38 个必要文件和 Git blob SHA，URDF 的 25 个不同网格引用齐全：[DOWNLOAD_MANIFEST.json](assets/sharpa_wave/official/DOWNLOAD_MANIFEST.json)。官方独立手模型的静态 USD 组合检查记录于 [ASSET_VALIDATION.json](assets/sharpa_wave/official/ASSET_VALIDATION.json)。

## UR5e 与 W01 的组合

构建器为 [scripts/build_sharpa_robot.py](scripts/build_sharpa_robot.py)，当前组合资产在 [assets/ur5e_sharpa_w01_right/](assets/ur5e_sharpa_w01_right/)。

- 28 个主动关节：UR5e 6 个、W01 22 个，保留官方手关节名称、轴、限位、惯量和源网格。
- 手掌参考系 `right_hand_C_MC`，手根 `right_hand_flange`；保留五指 DP、elastomer、fingertip 坐标关系。
- `tool0` 与手法兰间是本项目建模的圆柱适配器：长 20 mm、半径 30 mm，密度 2700 kg/m³。尚未验证实物孔位、电气或安装兼容性。
- USD 导入合并固定关节后为 29 个刚体、35 个碰撞体。手掌并入 `wrist_3_link`，指腹皮肤并入对应 DP 刚体；触觉接口用保存的 `DP_T_elastomer` 恢复皮肤坐标。
- 自碰撞开启，7 组机构过滤来自官方 USD，并适配合并后的名称。[过滤关系](analysis/sharpa_grasp/official_usd_collision_filters.json)
- 镊子和方块由任务场景创建，为动态物体，任务执行中没有把它们固定附着到机器人。

来源、输入哈希与安装变换见 [robot.build.json](assets/ur5e_sharpa_w01_right/robot.build.json)，导入拓扑见 [robot.import.json](assets/ur5e_sharpa_w01_right/robot.import.json)，关节和坐标系见 [wave01_interface.json](analysis/sharpa_research/wave01_interface.json)。

在新目录重建：

```bash
python3 scripts/build_sharpa_robot.py \
  --output assets/ur5e_sharpa_w01_rebuild --adapter-length .020
/cpfs/user/lvzhaoyang/apps/bin/isaac-python scripts/build_sharpa_robot.py \
  --output assets/ur5e_sharpa_w01_rebuild --import-only --device 0
```

构建需要外部 UR5e 源文件与本地 W01 原始资产。改变安装变换或碰撞模型后，需要重新验证握姿和稳定快照，不能直接继承旧结果。

## 两条触觉数据通路

| 接口 | 输出 | 含义与限制 |
|---|---|---|
| [src/sharpa_tacmap.py](src/sharpa_tacmap.py) | `tacmap.npz`、`tacmap.json` | 调用未修改的官方 `SharpaTacmap`；五指 240×240 的量化、模糊几何压入图。没有用力图生成灰度，不是 FEM、压力或光学 RGB 仿真 |
| [src/sharpa_tactile.py](src/sharpa_tactile.py) | `tactile.npz`、`tactile.json` | 独立读取 PhysX 每个物理子步的法向力、摩擦力及接触位置；另保留历史表面力投影。`N/bin` 未除以面积，不能称压力 |

TacMap 输入为指腹表面点/法线与实际目标位姿；目标是镊柄及两片镊臂的闭合网格。它不以仿真接触力作为输入。当前 CAD 派生地图仍是近似对应，并非真实硬件像素标定；原始官方地图和全零对照保留用于诊断。

PhysX 力读取器当前默认把表面采样成 60×60 格，仅用于历史力投影。超过映射距离阈值的接触仍保留在原始/刚体统计中，不能将图中缺失解释为无接触。固定关节合并后，DP 刚体统计可能包含皮肤以外的碰撞贡献，需要进一步做接触来源归因。

[sharpa_tacmap_compat.py](src/sharpa_tacmap_compat.py) 在当前 Lab 2.2.1 / Sim 5.0 环境中加载局部 2.3.2 接口，不覆盖系统包。此版本选择及当前刚性接触均不等于完整复现官方压入示例，具体差异见当前实验说明。

## 开发、检查与媒体工具

- 完整任务：[scripts/run_sharpa_w01.sh](scripts/run_sharpa_w01.sh)，默认配置 `configs/sharpa_w01_natural_v6d_tacmap_matched_v2.json`、稳定快照 `configs/sharpa_w01_natural_v6c_preheld.json`；通用运行器为 [run_pick_place.py](scripts/run_pick_place.py)。
- 持镊与闭合探针：[analysis/probe_straight_face_pinch.py](analysis/probe_straight_face_pinch.py)，接收 `--pose`、`--variants`、手驱动/摩擦和弹簧参数。短探针不能替代完整任务验收。
- 原始轨迹验收：[analysis/validate_pick_place.py](analysis/validate_pick_place.py)；官方 TacMap：[validate_sharpa_tacmap.py](analysis/validate_sharpa_tacmap.py)；独立接触记录：[validate_sharpa_tactile.py](analysis/validate_sharpa_tactile.py)。
- 源网格轨迹审计：[analysis/sharpa_grasp/audit_trajectory_mesh.py](analysis/sharpa_grasp/audit_trajectory_mesh.py)。`--stride 1` 检查全部 active 控制样本，仍不代表连续时间或全部物理子步。
- 当前视频：[scripts/compose_sharpa_tacmap_video.py](scripts/compose_sharpa_tacmap_video.py)；历史力图视频：[compose_sharpa_video.py](scripts/compose_sharpa_video.py)。相机、触觉和轨迹必须来自同次运行。
- 派生地图生成：[analysis/sharpa_tacmap/build_w01_matched_maps_v2.py](analysis/sharpa_tacmap/build_w01_matched_maps_v2.py)，使用保留的原始地图和当前 CAD；表面对应及掩码记录在派生地图目录中。

仿真使用 `/cpfs/user/lvzhaoyang/apps/bin/isaac-python`，CPU 几何和媒体工具使用 `.venv-human/bin/python`；依赖见 [analysis/sharpa_grasp/requirements.txt](analysis/sharpa_grasp/requirements.txt)。所有新实验使用独立输出目录。

历史 v5 结果保留于 [sharpa_w01_final.json](reports/sharpa_w01_final.json)，其姿态不同，不能与当前自然握姿混用。历史文字快照见 [整理前文档](reports/cleanup_20261020/readme_before/README_SHARPA_W01.md)，其中已删除的探索路径以 [清理清单](reports/cleanup_manifest_20261020.json) 为准。
