# 连续取镊与夹放演示

来自 2026-10-29 的固定案例运行 `tweezer_pipeline_calibrated_20261029_111556`。场景使用已标定的 D405 腕部相机和 D435 全局相机，支架与镊子沿世界 +Y 平移 10 cm。

[![完整流程，2 倍速](global_preview.gif)](global.mp4)

- [全局原速视频](global.mp4) / [腕部原速视频](wrist.mp4)：640×480、30 FPS，各 1354 帧。
- `global_preview.gif`：全局完整流程的 2 倍速预览，480×360、10 FPS、循环播放；没有截掉任务阶段。
- `evaluation.json`：固定案例成功 1/1，取镊约 21.57 秒，夹物与放置约 23.52 秒，最终 XY 误差约 0.62 mm。
- `scene_layout.json`、`camera_manifest.json`：实际场景与相机配置；`source_check.json` 为当次源码一致性检查。
- `media_verification.json`、`MANIFEST.sha256.json`：本演示文件的媒体核验、来源和哈希。

本次流程仍在取镊后返回 HOME 附近，再开始夹物；交接不重置机器人或物体。单案例成功不代表随机场景成功率，也不是真机自主操作结果。

原始轨迹、源码快照、检查点和逐步日志保留在本机 `simulation/runs/tweezer_pipeline_calibrated_20261029_111556/`，本演示目录只包含便于查看的媒体和结果摘要。重跑命令见 [根 README](../../README.md#快速运行)。
