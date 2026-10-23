# 已标定双相机 Demo

使用已确认的 D405 腕部 / D435 全局外参和 rectified 内参录制完整抓放场景。

```bash
source /mnt/workspace/tweezers-sim2real/env/setup_sim.sh
cd "$PROJECT"
RUN_DIR="runs/xhand_calibrated_demo_$(date -u +%Y%m%dT%H%M%S_%N)"
bash scripts/run_calibrated_demo.sh "$RUN_DIR"
```

输出在 `$PROJECT/$RUN_DIR`：

- `wrist.mp4`、`global.mp4`：640×480、30 FPS、H.264/yuv420p 两路视频。
- `wrist_first.png`、`global_first.png`、`*_last.png`：首尾无损截图。
- `camera_frames.jsonl`：每个视频帧的仿真时间、关节状态、TCP 和两路光学坐标系世界位姿。
- `camera_manifest.json`：标定配置、内参、基座变换、图像模式和输入哈希。
- `evaluation.json`、`trajectory.npz`：原任务成功判定与轨迹。
- `scene_layout.json`：实际桌面尺寸、中心、HOME、工具偏航及物块初始/目标位置。
- `launch.json`、`source/`、`execution_source_provenance.json`：运行参数和来源记录。

默认使用根目录 `models/policies/xhand_pick_place_ppo.pt`、`models/snapshots/xhand_stable_grasp.json`，单案例、seed=0、retreat_closure=0.8。正式相机参数在 `calibration/results/`，适配器在 `calibration/examples/`。每次输出目录必须不存在。原有 `--camera-config` 模式继续使用原观察相机；此入口通过 `--calibrated-rig` / `--world-base` 使用已确认适配器。

默认布局配置为 `configs/calibrated_demo_y_table.json`：桌面 X×Y×Z 为 `0.80×1.10×0.04 m`，长边沿世界 Y 轴；桌面中心 XY 为 `(0, 0.35)`，表面高度仍为 `0.72 m`。物块案例在 `configs/calibrated_demo_y_case.json`：初始 XY `(0.22, 0.48)`，目标 XY `(-0.08, 0.76)`，物块中心高度由任务按桌面计算。

HOME XY 改为 `(0, 0.48)`，保存的稳定手/镊子抓握整体绕世界 Z 轴旋转 90°，通过已有 reset IK 重新求机械臂关节角。桌面、HOME、工具朝向和案例相当于旧布局绕机器人基座旋转 90°，不是旋转机器人基座。旧稳定快照作为来源保留。这里是便于双相机观察的近似布局，没有使用真实桌面测量值。

相机保持真机标定机位，腕部每次状态更新后随 wrist_3_link 更新，全局相机固定。控制器、原 PPO checkpoint、摩擦、质量和接触参数沿用 demo。新的世界坐标位置和姿态不保证 PPO 具有旋转等变性，需以该布局的完整回合验证结果判断。原脚本不传 `--scene-layout` 时继续使用旧布局默认值。

仅在仿真中运行，不连接真实机械臂或 RealSense。输出通过已验证 `rig.rgb()` 重采样到 rectified K；底层相机渲染图像是中间结果。

快速检查启动/录像可追加 `--episode-seconds 0.5`；这只测试短时运行，不构成完整任务成功验收。

2026-10-23 已完整验证新 Y 轴桌面布局，结果在 `runs/xhand_calibrated_y_table_20261023`：seed=0 的指定案例成功 1/1，最终 XY 误差约 0.434 mm，仿真时长约 23.70 s。两路 H.264 视频均为 712 帧、640×480、30 FPS，已逐帧解码检查；来源边界检查一致。详情见该目录的 `demo_verification.json`。这是该单案例的运行结果。
