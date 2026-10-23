# 项目维护约定

- 当前主线是 UR5e + XHand 使用预持镊子抓放物块；根 README、PROJECT_PROGRESS.md 和 scripts/run_demo.sh 为当前入口。
- calibration/ 为真实相机标定工具，simulation/ 为 Isaac Sim 任务；两者使用独立 Python 环境，不为离线标定修改 Isaac Sim 环境。
- models/ 保存运行必需的冻结策略和稳定抓握快照，不是运行缓存；simulation/runs/、calibration/data/ 和 verification/ 为本地数据与结果。
- archives/calibration_confirmed_20261023 为冻结证据，保持其文件与清单原样。重跑检查写入新的 verification/ 目录。
- 修改使用仓库相对路径；原始采集 JSON、历史运行日志中的绝对 source 路径是溯源元数据，不做全局替换。
- 用户要求更新进度时，更新 PROJECT_PROGRESS.md 的实际完成状态、验证指标与输出路径；后续尚未完成阶段不提前填写成果。
- 当前双相机联合交叉验证暂缓。单案例成功不能写成随机总体成功率，仿真误差不能写成真机精度。
- 不在仿真运行期间修改其已跟踪的源码或配置，运行器会检查来源边界哈希。
