# 灵巧手使用镊子夹取、搬运和放置方块

当前工作版本为 **UR5e + Sharpa Wave W01 v6d 自然握姿 + 官方 TacMap 算法**。更新日期：2026-10-20。

拇指和食指夹住镊臂较厚的宽面，中指在下方提供备用承托、释放时略微退让，无名指和小指弯曲收拢。任务从已持镊的稳定初态开始，执行接近、夹起、搬运、放下和撤离。使用几何阶段控制器，**尚未训练 W01 专用 PPO**。

当前成果是可复现的物理夹放与几何触觉演示。原默认入口保留刚性接触；新增 [柔顺接触实验入口](scripts/run_sharpa_w01_compliant.sh)使用 280/20 皮肤材质、重新平衡的持镊与闭合控制。**没有真实传感器标定，也没有将运动学压头的整套偏移参数直接用于动态镊子。**

最新改进完整案例通过，提升 **83.97 mm**、落点误差 **1.153 mm**；同八组起终点回归 **5/8，未达到原刚性8/8**。还发现官方多目标CPD会跨两片镊臂误接受盒外点，已保留原图并加入独立逐目标复算。新入口属于实验版本，详见 [最新结果与复现](README_SHARPA_W01_NATURAL.md)、[改进报告](sharpa_w01_tactile_validation/improvement_validation/IMPROVEMENT_VALIDATION_REPORT.txt)。以下旧成绩和待改进项主要描述刚性基线。

## 当前进展与证据

| 已完成内容 | 结果与范围 | 证据 |
|---|---|---|
| 官方 W01 资产接入 | 右手 22 个主动关节，与 UR5e 组合；保留上游来源和许可证 | [资产与接口说明](README_SHARPA_W01.md) |
| 自然握姿及夹放 | v6d 新随机案例 **6/8**，两例在抬升、搬运阶段失败；旧坐标回归 **8/8** | [自然握姿最终报告](reports/sharpa_w01_natural_v6d_final.json) |
| 官方 TacMap 完整录像 | 固定案例 **1/1**，25.63 s；最大抬升 **84.21 mm**，最终落点误差 **1.00 mm** | [最终报告](reports/sharpa_w01_official_tacmap_final.json) |
| 五指触觉与独立接触力 | 五指各 **240×240、770 帧、30 Hz**；另保留 **480 Hz** PhysX 法向及摩擦接触记录 | [最终运行目录](runs/sharpa_w01_official_tacmap_matched_v2_video/) |
| 算法与几何校验 | 官方算法未改；两套表面的固定几何对照、实际任务抽样射线复算通过；固定几何对照不等于材料标定 | [详细实验、验证与限制](README_SHARPA_W01_NATURAL.md) |

官方 TacMap 录像的物理轨迹与原 v6d 录像相同；增加传感器没有重新评估随机成功率。不能把单例 1/1、回归 8/8 和新随机 6/8 合并成通用成功率，也不能使用历史 XHand 的 64/64 作为 W01 成绩。

直接查看媒体：

- [完整仿真与官方 TacMap 视频](outputs/sharpa_w01_official_tacmap_matched_v2.mp4)
- [触觉与独立接触力响应曲线](outputs/sharpa_w01_official_tacmap_response.png)
- [代表帧：搬运](outputs/sharpa_w01_official_tacmap_matched_v2_frame_00462.png)、[代表帧：撤离](outputs/sharpa_w01_official_tacmap_matched_v2_frame_00712.png)
- [触觉几何正反例图](outputs/sharpa_w01_official_tacmap_matched_controls.png)

## 当前问题

1. **触觉图与力的语义不同。** PhysX 求接触力，Isaac Lab 提供读取和射线接口，TacMap 根据实际位姿、表面点和法线计算几何压入，再量化、模糊。灰度不是压力、力或经过标定的真实硅胶形变量。
2. **接触物理尚未与官方示例对齐。** 当前皮肤和工具的 `contact_offset=0.2 mm`、`rest_offset=0`，未启用柔顺接触。官方示例皮肤刚度/阻尼为 `280/20`，皮肤偏移为 `0.2/−0.1 mm`，运动学压头为 `0.1/−0.5 mm`。这些是示例值，不是任意动态镊子任务的最佳参数；接触柔顺仍是刚体弹簧阻尼近似，不是 FEM。
3. **食指响应弱且局部。** 全录像食指峰值灰度仅 5；15.42 s 附近最大几何压入约 0.0112 mm，仅 17 个采样点为正，量化峰值 2 经平滑变为 1。低信号在平滑前已经存在，不能据此判断食指没有夹持力。[低灰度诊断](analysis/sharpa_tacmap/index_low_gray_diagnosis.json)
4. **触觉表面和碰撞代理仍有差异。** 当前地图由 W01 CAD 几何匹配派生，并非原始硬件 taxel 标定。原版地图在此模型上的全零对照保留。中指有 14 帧出现几何灰度、同帧全部物理子步却无接触力；皮肤并入末节指骨刚体后，受力来源还需细分。[力与几何对齐记录](runs/sharpa_w01_official_tacmap_matched_v2_video/tacmap_force_alignment.json)
5. **任务和真实性边界。** 尚未解决所有随机案例；不包含从桌面捡起镊子、视觉闭环、未知物体或实机迁移。已有静态/离散源网格审计不等于连续时间无穿透证明，触觉算法校验通过也不等于物理材料正确。

## 后续改进顺序

1. 在独立输出目录复现官方配套模型、原始地图、柔顺材料及压头两侧偏移的加载/卸载基准。既有固定几何测试只验证算法；此前 3 秒局部柔顺探针也不是完整任务对照。
2. 检查 PhysX 实际碰撞代理、指腹触觉有效区和接触位置，区分皮肤与指骨承力，确保食指以合适的指腹区域夹持镊子厚侧。
3. 以官方 `280/20` 为候选做皮肤柔顺性和偏移敏感性实验，验证受控压入响应、离开归零及数值稳定性；不直接把压头的 `−0.5 mm` 静止偏移照搬到 1 mm 厚镊臂。
4. 在新物理参数下重新形成预持镊平衡，调整闭合、持握和释放，再按原成功条件复测完整夹放及独立随机案例，保留失败结果。
5. 同时记录量化前深度、原始灰度、接触力和承力位置；需要放大显示时明确标注量程。获得实机力—压入—触觉数据后再进行定量标定。

## 快速复现

在项目根目录执行。当前运行环境是 **Isaac Lab 2.2.1 / Isaac Sim 5.0**；TacMap 所需缺失接口由项目内固定版本的 **Lab 2.3.2 兼容代码**提供，没有覆盖系统安装。已核查的 Sharpa TacMap README 未明确锁定 Isaac 版本；2.3.2 是兼容代码来源，不是 Sharpa 官方推荐版本。

```bash
# 单例夹放：默认 W01、自然握姿、匹配地图配置和稳定快照。
scripts/run_sharpa_w01.sh --out runs/w01_new_eval \
  --num-envs 1 --evaluate zero --device cuda:0

# 同次运行记录原生相机、官方 TacMap 与独立 PhysX 接触力。
scripts/run_sharpa_w01.sh --out runs/w01_new_tacmap \
  --num-envs 1 --evaluate zero --device cuda:0 \
  --cases configs/sharpa_w01_hard_case.json \
  --camera-config configs/sharpa_w01_tacmap_cameras.json \
  --tactile --tacmap

# 已有数据的独立验证；更完整的检查和媒体合成命令见详细实验说明。
python3 analysis/validate_pick_place.py runs/w01_new_tacmap --geometry
python3 analysis/validate_sharpa_tacmap.py runs/w01_new_tacmap \
  --require-video --require-task-success \
  --controls runs/sharpa_official_tacmap_controls_matched_v1
```

每次使用新的 `--out`；入口拒绝覆盖已有证据。`--device cuda:N` 选择 GPU，不额外设置 `CUDA_VISIBLE_DEVICES`。同种子跨设备/求解器不保证逐位一致；重新运行必须独立验收。W01 使用 `--evaluate zero` 名义阶段控制器，通用 `run_pick_place.py` 仍保留 XHand 默认值，请优先使用 W01 专用入口。

## 代码与文档导航

| 位置 | 用途 |
|---|---|
| [README_SHARPA_W01_NATURAL.md](README_SHARPA_W01_NATURAL.md) | 当前握姿、官方 TacMap、实验结果、问题与完整复现步骤 |
| [README_SHARPA_W01.md](README_SHARPA_W01.md) | W01 资产来源、许可证、模型构建和触觉接口 |
| [README_XHAND.md](README_XHAND.md) | 独立保留的 XHand v15 基线、模型、快照与复现 |
| [scripts/run_sharpa_w01.sh](scripts/run_sharpa_w01.sh) → [run_pick_place.py](scripts/run_pick_place.py) | W01 参数入口与共用任务运行器 |
| [src/rl_scene.py](src/rl_scene.py)、[pick_place_task.py](src/pick_place_task.py)、[rl_task.py](src/rl_task.py) | 场景、阶段控制、抓放与持镊判据 |
| [src/sharpa_tacmap.py](src/sharpa_tacmap.py)、[sharpa_tacmap_compat.py](src/sharpa_tacmap_compat.py) | 官方几何触觉适配及版本兼容 |
| [src/sharpa_tactile.py](src/sharpa_tactile.py) | 独立 PhysX 法向力、摩擦力、接触点记录及历史力图 |
| `analysis/` | 保留的独立验收、几何诊断、地图构建和关键对照工具 |
| `assets/`、`configs/` | 官方/组合资产、原始/派生触觉表面、已选配置和稳定快照 |
| `runs/`、`reports/`、`outputs/` | 原始实验与关键历史证据、结论和清理记录、可直接查看的媒体 |

## 整理与证据保留

2026-10-20 整理将当前 W01 说明放在主 README，XHand 历史基线独立为 `README_XHAND.md`。按依赖清理已被替代的探索代码、候选配置、中间结果、重复场景和交付压缩包；保留当前复现链路、失败案例、最终媒体、官方原始资产、许可证与关键对照。

本次实际删除 **351 个文件，7,560,904,624 字节（约 7.56 GB / 7.04 GiB）**，其中包含 11 个过时代码文件、6 份未采用配置和 219 个 Python 缓存文件。710 个受保护文件清理前后校验一致。旧开发运行的轨迹、配置、源码和报告仍保留，但已删除其大型 USD 导出；重复这些旧运行的 USD 专项审查需要重新导出场景。当前 W01 和 XHand 最终验收场景完整保留。

实际删除项、原因和保留文件校验见 [本次清理清单](reports/cleanup_manifest_20261020.json)，整理后的检查见 [清理验证](reports/cleanup_validation_20261020.json)。清理清单是审计记录，不是备份。历史报告可能引用已清理的探索路径，应结合清单理解；当前文档中的复现路径须可用。

冻结实验报告保持原样，其中 README 哈希对应当时文档；本次更新前的文档保留于 [readme_before](reports/cleanup_20261020/readme_before/)。本次整理不重跑仿真、不改变已有成功/失败结果，也不表示上述物理问题已经解决。`runs/` 仍含 XHand 模型和稳定快照，不能整目录作为缓存删除。
