# UR5e + XHAND1 右手仿真模型

`robot.urdf` 是组合模型，单位为 m / kg / rad。独立源模型也保存在 `ur5e.urdf`、`xhand_right.urdf`；它们使用同一份相对网格路径。源文件保存在 `sources/`，获取与组合过程由 `scripts/fetch_ur5e_xhand.py` 重现。

此目录的静态审计已通过，但静态文件可解析不代表 P0/P1 动力学验收通过。物理验收状态以项目阶段报告为准。

## 版本与来源

| 资产 | 固定源版本 | 说明 |
|---|---|---|
| UR5e | `UniversalRobots/Universal_Robots_ROS2_Description` commit `89bbe795f38a7ab00fb66fe8831dfff79dc99edf` | 官方 UR5e YAML 参数通过官方 xacro 展开；未使用 UR5/UR10 缩放替代。 |
| XHAND1 右手 | `facebookresearch/spider` commit `71238456bf97a7eeb3d0471aa31974e2d404d4ae` | 研究模型，原始 URDF blob 为 `b833f40d4598b149d4f544ad9be06097cd264c33`；非厂商标定文件。 |
| 转接件 | 本项目解析几何 | 半径 30 mm、厚 20 mm 的实心铝圆柱；密度 2700 kg/m³，质量约 0.15268 kg，惯量由几何积分公式计算。 |

每个源文件的 SHA-256 和大小见 `provenance.json`，生成 URDF 也有独立哈希。原始 URDF、xacro、YAML、STL/DAE 不改写；组合时只修正引用路径、世界固定结构与标记实体，并加入明确的仿真转接件。

UR5e 内容依照源仓库声明为 BSD-3-Clause（该仓库特殊网格条款不涵盖 UR5e）；SPIDER 仓库声明为 CC-BY-NC-4.0。许可证原文与来源 README 一并保存。此处未把第三方研究资产重新声明为本项目许可证。

## 坐标、安装与关节

- 根：`base_link`；机械臂法兰：`flange`；UR 零工具坐标：`tool0`。
- 掌根：`right_hand_link`，源模型中手掌和手指从掌根向 `-Z` 延伸。
- 掌内参考点：`right_hand_ee_link`，位于掌根 `(0, 0, -0.065)` m。
- `T_tool0_palm`：平移 `(0, 0, 0.020)` m，旋转 RPY `(π, 0, 0)`。`T_flange_palm` 同时保存到机器人 JSON 配置；两个坐标的轴约定不能混用。
- 转接件近端与 `tool0` 原点重合，远端接手腕。这个安装是明确的仿真设计，未验证真实螺孔、紧固件、电气接口和制造尺寸。
- UR5e 原始 `world` 和 `base_joint` 已删除。组合 URDF 不含世界锚点，Isaac 导入时通过 `fix_base=True` 固定一次。桌面位置由场景设置，手掌不固定到世界。

六个机械臂关节顺序：

```text
shoulder_pan_joint shoulder_lift_joint elbow_joint
wrist_1_joint wrist_2_joint wrist_3_joint
```

十二个手关节顺序：

```text
right_hand_thumb_bend_joint
right_hand_thumb_rota_joint1
right_hand_thumb_rota_joint2
right_hand_index_bend_joint
right_hand_index_joint1
right_hand_index_joint2
right_hand_mid_joint1
right_hand_mid_joint2
right_hand_ring_joint1
right_hand_ring_joint2
right_hand_pinky_joint1
right_hand_pinky_joint2
```

全部为独立 revolute 关节，无 mimic、自由腕或额外可控安装关节。完整限位、速度与力矩上限采用源文件值，见 `configs/robots/ur5e_xhand_right.json`；手部值仍需物理验证，不声称是实机已标定值。

## 固定标记与惯量

源 XHand 有六个质量约 `3.5343e-8 kg` 的 EE/tip 固定坐标标记。它们的小型几何是坐标标记，不作为独立指腹或刚体使用：组合模型保留其固定变换，移除标记的 visual/collision/inertial。十三个真实手部 link 的几何与惯量保留，手部总质量约 `1.12238484 kg`。

Isaac 使用 `merge_fixed_joints=True`，将掌部、转接件与腕部合成同一物理刚体，将末端标记并入相应指节，不任意放大微小质量。`fixed_frame_map` 记录每个固定 frame 对应的动态 body 与精确变换。

导入脚本会额外创建无刚体、无质量的 `__frames/<frame>` Xform。`robot.import.json` 中的 `frame_paths` 是导入后坐标路径的权威映射；`rigid_body_paths` 用于接触与刚体状态。掌部物理接触归属于合并后的 `wrist_3_link`，不能把 marker Xform 当作独立接触刚体。

原始 UR 的基座质量在官方配置中也标注了不确定性；本模型保留该值。固定底座的总质量不是末端负载。组合模型总质量约 `22.97506624 kg`。

## 复现

构建只需独立 Python 环境中的 `xacro==2.1.1` 与 PyYAML；不修改 Isaac Sim Python 依赖。示例：

```bash
python -m venv /tmp/ur5e-xacro-venv
/tmp/ur5e-xacro-venv/bin/python -m pip install \
  --index-url https://mirrors.aliyun.com/pypi/simple/ xacro==2.1.1 pyyaml==6.0.2
/tmp/ur5e-xacro-venv/bin/python scripts/fetch_ur5e_xhand.py
```

脚本先直连 GitHub，下载失败后才使用当前服务器已有的 `with-general-proxy`；不会读取或显示代理凭据。可通过 `--ur-source` / `--hand-source` 使用已经下载且无修改的固定 commit 源仓库。

在可用 GPU 上单独启动导入：

```bash
/cpfs/user/lvzhaoyang/apps/bin/isaac-python scripts/import_ur5e_xhand.py
```

产出 `robot.usd` 和 `robot.import.json`。导入脚本不播放物理时间线。它启用源碰撞网格的 convex decomposition、单一固定基座、重力、自碰撞、32/8 求解器迭代和有限力矩 force drives。刚体相邻关节过滤遵循 PhysX 规则，非相邻碰撞保留。所有这些设置仍需 P0/P1 实测。

导入器默认创建的实例化网格子树会被显式去实例化，确保碰撞数量、接触参数和各指材料能够逐项读取／设置；不能把默认遍历看不到的实例代理误报为“没有碰撞形状”。每个真实动态 body 必须有至少一个 collider，否则导入脚本报错。

手部采用 128 个凸包上限、每包 64 顶点、2,000,000 体素分辨率、0.5% 误差、0.1 mm 最小厚度和 shrink-wrap。这些值针对毫米级指根凹槽；合并到 `wrist_3_link` 下的掌网格也应用相同设置。默认较粗分解在首次 P0 检查中产生掌—食指根及掌—拇指根假接触；细化碰撞几何后短时诊断已消失，未过滤这些碰撞对、缩小手或扩大间隙。

配置中的位置驱动增益以 Nm/rad 与 Nm·s/rad 表示；写入 USD angular drive 时转换为每 degree 单位。后续直接通过 Isaac articulation 设置增益时，应使用该接口自身的 rad 单位，避免重复换算。

## 本次静态检查与待验证点

已检查：一个根节点、无重复父子关系、18 个主关节、无 mimic、所有 mesh 引用存在、所有真实惯量正定、主惯量满足三角不等式、物理碰撞 link 具有质量。详细列表见 `model_audit.json`。基座碰撞几何最低点与 `base_link` 平面相差约 `1e-7 m`，桌面初始摆位采用同一平面。

尚需由 P0/P1 验证：实际 USD 导入与质量合并、所有手指碰撞分解覆盖、静止 10 s、关节逐个驱动、接触 link 归属、末端 FK 与 PhysX 一致性、非预期自碰撞、基座与桌面相接处过滤及初始位置的可达性。源手部 `dynamics` 参数属于第三方模型初值，不能直接当作真实摩擦或阻尼标定。

`source_mesh_contact_audit.json` 记录了 CPU FCL 精确三角网格诊断：手关节全零时，掌—食指第一节的真实表面间隙为 2.30270 mm，掌—拇指第一节间隙为 3.26038 mm；两对均无源三角表面相交。单凸包会在掌—食指处产生伪相交。食指第一节 STL 存在非闭合边，此项不能当作完整体积证明；实际物理验收仍是必要证据。该检查通过 `scripts/audit_dex_mesh_contacts.py` 重现，不更改源网格。

`scripts/plan_dex_contact_probe.py` 使用真实末节 STL 面与法向布置五个独立 4 mm 半径固定诊断球，FCL 检查初始分离与非目标指扫动间隙，输出 `reports/ur5e_xhand/p1_contact_plan.json`。诊断球是测试夹具，手指触碰仍通过有限驱动执行。计划中的几何接触预测不能替代 P1 实际接触回调、法向、退出与误触验收。
