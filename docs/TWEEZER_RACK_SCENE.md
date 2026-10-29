# 镊尖朝下的取镊初始场景

这是独立的场景预览入口，复用 UR5e + XHand 和原有 18 cm 被动弹簧镊子。
支架为固定碰撞体，镊子为受重力作用的动态刚体，当前通过顶部托舌承托，四面定位槽壁已去除；没有把镊子固定到支架的关节或吸附力。

在仓库根目录运行（输出目录必须不存在）：

```bash
source env/setup_sim.sh
"$ISAAC_PYTHON" scripts/render_tweezer_rack.py \
  --out runs/tweezer_rack_preview_new --device cuda:0
```

配置：`simulation/configs/tweezer_rack_scene.json`。尺寸与坐标单位为米。
桌面高 0.72 m，镊尾根部位于 `[0.20, 0.52, 1.00]`，镊尖约在桌面上方 0.10 m。
支架底座 120 × 140 mm；4 mm 宽的托舌进入两镊片之间承托连接尾部。
顶部没有定位槽壁，设计取出方向是先向上抬升脱离托舌，再远离支架。该取出路径尚未做机器人抓取验证。

朝向配置：`tool_yaw_deg=180` 将镊片开合轴设为世界 −X。
`rack_yaw_deg=0` 独立控制支架，相对 `tweezer_rack_aligned_20261028` 预览旋转 180°，镊子和机械手朝向保持不变。
`rack_guides=false` 不创建四面定位槽壁及其碰撞体；可设为 true 恢复槽壁作对照。
机械手参考方向取自稳定持镊快照，并按当前桌面任务绕世界 Z 旋转 90°。
架上镊子保留竖直朝下，只匹配开合主轴；不复制快照中约 30° 的镊身倾斜。
`scene_report.json` 记录实际开合轴与参考轴的误差，此配置不是精细握姿或可抓性验证。

输出：

- `overview.png`：机械臂、桌面与支架的展示全景。
- `rack_closeup.png`：支架与镊子的展示近景。
- `scene_report.json`：配置、支架尺寸、3 秒稳定过程后的实际状态与位置保持检查。
- `settling_trajectory.npz`：镊尾和两侧镊片的逐物理步状态。
- `scene.usd`：用于检查布局的 USD 导出。Fabric 动态状态不保证同步回 USD；重现物理初态应运行脚本，测量状态以轨迹为准。
- `camera_manifest.json`、`views.json`：展示相机参数，不代表已标定的训练相机。

本入口不加载 PPO、不执行取镊、不启动训练。`stable_on_rack` 仅表示短时位置与朝向保持通过，不代表接触抖动、可抓性、脱架轨迹或后续技能交接通过验证。
