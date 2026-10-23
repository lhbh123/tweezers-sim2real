"""Build the final Chinese text/PDF report from completed recorded evaluations."""
from pathlib import Path
import json
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/rl_scene.py').is_file())
BASE=ROOT/'sharpa_w01_tactile_validation/improvement_validation'
font_manager.fontManager.addfont(str(BASE/'assets/NotoSansSC-Regular.ttf'))
plt.rcParams.update({'font.family':'Noto Sans SC','axes.unicode_minus':False,'pdf.fonttype':42})


def main():
    compare=json.loads((BASE/'visualizations/task_improvement_comparison.json').read_text())
    final=compare['runs'][-1]
    batch=json.loads((BASE/'runs/compliant_v1_replay8/evaluation.json').read_text())
    replay=json.loads((BASE/'per_target_replay/full_trajectory/report.json').read_text())
    replay_peaks=[replay['summary'][n]['min_gray_peak'] for n in replay['finger_names']]
    trial_lines=[]
    for name in ['soft_original_control','rebalanced_task_fast','rebalanced_cl165_fast',
                 'rebalanced_cl175_fast','prehold_ext025_fast','prehold_ext025_cl200_fast',
                 'prehold_ext025_cl250_fast','compliant_v1_full_validation_v2']:
        p=BASE/'runs'/name/'evaluation.json'
        if p.exists():
            r=json.loads(p.read_text())
            trial_lines.append(f"{name}: {r['successes']}/{r['episodes']}；最大提升 {r['max_lift_m'][0]*1000:.3f} mm；最高阶段 {r['max_phase'][0]}；最终误差 {r['final_xy_error_m'][0]*1000:.3f} mm。")
    text=f'''Sharpa W01 镊子任务：触觉与控制improvement_validation
日期：2026-10-20

结果
完整记录案例：{'通过' if final['success'] else '未通过'}。最大提升 {final['max_lift_mm']:.3f} mm，最终 XY 误差 {final['final_xy_error_mm']:.3f} mm，持续 {final['duration_s']:.3f} s。
相同八组起终点回归：{batch['successes']}/8。对应旧基线为 runs/sharpa_w01_natural_v6d_regression 的8/8；不是另一次随机seed的6/8。控制、接触材料及预持镊同时升级，不能把成功率差异单独归因于材料。
版本状态：新柔顺入口属于实验版本。当前任务鲁棒性尚未达到旧刚性基线，不替换原默认入口。已经改善接触模型和观测可信度，不表示全部操作问题已经解决。
完整记录的五指原灰度峰值：{final['gray_peaks']}（拇、食、中、无名、小指；固定0—255）。其中食指高灰度包含已确认的跨镊臂射线误接受，不代表正确压入，也不作为“越亮越真实”的评分。
提起/搬运阶段宽面判据成立比例：{final.get('face_fraction_lift_transfer')}。该比例与双镊尖持续夹块是不同指标；不会把单例任务通过描述为全程严格宽面夹持。

已落地的改进
1. 力投影与TacMap显式使用同一个matched_v2表面，记录源文件哈希并断言一致。原始接触力不变；分别保留向量合力、模长载荷和端点/区间均值。
2. 从官方正在执行的两次射线和量化过程直接旁路记录原始CPD、首/反向距离、模糊前量化值。没有增加传感器更新次数、没有改变官方算法、没有增亮归一化。
3. 显式使用皮肤柔顺280/20、contact/rest=0.2/0 mm，镊子rest=0。保留原摩擦1.4、镊子弹簧5、手PD100/1、1/480s和真实逐关节力矩上限。280/20是官方基线参数，不是已标定的真实W01材料常数。
4. 重新建立预持镊平衡，更新有界闭合协同；open/closed/retreat=1/2.5/1。拇指IP在最大闭合目标仍保留1 mrad下限余量，其余关节承担额外闭合。没有放宽URDF限位。
5. 起抬前真实双方镊尖力各至少0.0604 N并持续0.25 s，同时保留原等待时间。该阈值来自2g方块、名义摩擦0.65和四倍理想静态余量；它是控制门槛，不保证运动中不滑脱。
6. 对负分离接触点采用严格验证的工具表面witness，仅在候选点确实落于已知外平面20µm内等条件成立时启用。原始点保留，宽面尺寸/法线/力阈值不放宽；法向力矩保持不变。这是测量分类修正，不是施加额外力。
7. 加入实际材质读回、原生碰撞表示导出、源码边界哈希和接触报告可用性记录。读取材质的第三列-280是柔顺刚度编码，不能解释为负恢复系数。

最终任务暴露的多目标CPD问题及处理
标准TacMap验证通过，只说明官方代码、调用过程与编码一致；不能保证这个多物体场景的输出都是正确形变。正式v2搬运13.3167s的食指有11232个正深度点，8337个在所有目标盒外超过1µm，仍由跨镊臂正反射线误接受。峰值点距任何目标盒4.182mm；正射线命中Right约4.714mm，反射线命中Left约8.809mm而Right不命中。GPU/独立CPU在该峰点相差仅0.029µm，因此不是浮点边界噪声。
已增加独立逐目标复算：每个刚体分别用官方正反射线判据，保留每目标深度，再取最小有效正深度，并保存max合并敏感性结果。该帧食指正点11232→2893，灰度峰值240→153；拇指峰值113保持。该方法是场景适配扩展，离线使用已保存的真实运动位姿，不修改原始仿真或官方tacmap.npz，也不冒称原版官方输出。仍记录单body非凸形状的盒外误接受和边界不确定性，不能当作硬件形变标定。
完整758帧×5指逐目标复算已完成，五指峰值为{replay_peaks}。食指全程峰值241→155，几何CPD峰4.744→2.931mm；不是等量法向材料压缩。全轨迹每body正CPD点均位于该body至少一个源盒内，min/max合并逐像素相同，原始输入NPZ前后哈希相同。这里使用独立解析射线、浮点位姿重建和官方编码规则，不是新物理仿真或原GPU传感器张量。非凸/不连通body与超薄目标的通用局限保留在工具说明中。
最终原图抽样39个指帧中有11个未通过独立解析几何的严格1µm/1码阈值；不能宣称全帧独立几何复核全部通过。758×5个官方内部链路则逐值满足原CPD和量化规则，独立模糊误差≤1码。两种检验回答的问题不同；详见observation_checks和per_target_replay。
11个失败指帧实际涉及12个超阈值像素：7个起点距盒面仅0.004—0.044µm，发生入口/出口分支切换；5个为近切线点，距离误差1.10—2.19µm。与float32世界坐标舍入敏感性一致，但没有保存原GPU射线，不能称逐射线精确证明。严格阈值仍未通过；代表帧399本身通过该阈值，毫米级跨体误接受结论不依赖上述边界像素。

可识别的单镊臂2×2对照
同一W01、单镊臂小试件、摩擦/控制/步长/地图/碰撞近似固定。四组均稳定完成接近—释放并回零。
约20 mN实际接触帧：刚性rest0：20.23mN / CPD21.26µm / 灰2；柔顺rest0：19.61mN / 61.71µm / 灰10；刚性rest−0.1mm：20.22mN /121.35µm/灰22；柔顺rest−0.1mm：19.61mN/161.59µm/灰30。
因此负rest偏移本身即可明显增亮；主任务选择rest0，以免把偏移带来的几何重叠误当材料真实性。试件为1×14×20mm、6g的单轴引导小镊臂，不能替代180mm完整镊子的动力学。
全部1696帧独立解析盒射线校验：CPD最大误差0.173µm，灰度最大误差1码。原始GPU中间量与官方量化关系另有逐帧报告。

预持镊与接触几何
新选定快照冷启动3秒通过：末1秒位置极差≤0.575µm，归一化姿态变化0.00121°，有效双面接触100%，拇/食广面向内法向力约0.298/0.265N。末帧原灰107/103，CPD峰约0.731/0.617mm。
较早cold_soft_rest0_v3的食指接触点均在皮肤CAD的20µm邻域内，未发现整体法线反向。此为该冷启动的几何证据；不能据此给所有任务帧的皮肤/壳体力做精确归属。
当前GPU路径下原生get_contact_report连续返回空header/data，但两个Tensor力读者作用反作用误差约6e−8N。空报告表示观测不可用，不能表示无接触。正式图表继续将该力标为“合并DP载荷”。
当前stage原生凸分解查询成功（10个collider全部有效）。较早cold_v3食指触点到cooked皮肤中位0.242µm、最大3.20µm，到cooked壳体中位约6.55mm；中指触点到cooked壳体中位0.028µm、到cooked皮肤约1.46mm，支持该冷启动中指为壳体区域接触。跨run本体坐标比较不能替代原生shape ID。全matched_v2采样点沿法线贯穿线取碰撞并集最外出口：食指绝对偏移P95约404µm、拇指约595µm，边缘极值约1.54/4.38mm；这不是欧氏Hausdorff或当前接触区域误差。未将凸包替换或重采样冒充已校准修复。v1/v2导出的1292个几何数组逐值一致。该接口是当前stage的representation，不是resident PxShape直接内存dump；细节见observation_checks。

保留的失败试验
'''+'\n'.join(trial_lines)+'''
旧柔顺1.75方案在约11.0秒搬运中先出现方块相对下滑、夹力下降，随后双侧接触消失；固定时间转阶段没有主动张开手指。初始力门控的反事实也会通过，因此本次改进同时重建了预持镊和平衡闭合，而不是只加门控。
八例回归的三个未通过案例：例2在squeeze阶段超时，曾短时有足够夹力，但在至少1秒等待条件成立后持续时间不足0.25秒；例8曾抬升65.32mm，之后掉回桌面并在lift阶段超时；例7到45秒末XY误差13.82mm且宽面判据为false，未通过最终稳定放置。原始成功序列与独立逐步重算完全一致。不能通过放松验收或降低夹力门槛将这些案例改记成功。
探索性2.0/2.5快测期间，主源码在其完成初始化之后合入了可选门控，结束哈希因此显式标记漂移；归档启动源码和更新事件保留在review/controlled_source_update.json。最终完整验证v2在源码冻结之后启动，检查其execution_source_provenance.json。v1高开销诊断运行被主动中断，已保存中断原因和部分几何记录，不作为完整任务证据；v2保留完整480Hz原始接触数据、30Hz触觉图和相机录像。

复现
scripts/run_sharpa_w01_compliant.sh --out runs/NEW_W01_COMPLIANT \
  --cases configs/sharpa_w01_hard_case.json --device cuda:7 \
  --camera-config configs/sharpa_w01_tacmap_cameras.json \
  --tactile --tacmap --tacmap-diagnostics \
  --collision-geometry --contact-shapes
输出目录必须不存在。旧刚性入口scripts/run_sharpa_w01.sh保留。新配置和快照位于configs/sharpa_w01_tacmap_compliant_v1.json与configs/sharpa_w01_tacmap_compliant_preheld_v1.json。
逐目标对照：.venv-human/bin/python sharpa_w01_tactile_validation/improvement_validation/scripts/replay_per_target_tacmap.py runs/NEW_W01_COMPLIANT --output runs/NEW_W01_COMPLIANT_per_target

适用边界
从已有预持镊快照经动力学重新平衡起步，尚未实现从张开的手或桌面自主拿起镊子。匹配地图是当前W01 CAD派生映射，不是硬件像素标定。原生皮肤/壳体shape归属仍不可得。
正式图保留官方原始输出，独立歧义mask及逐目标复算单独保存。旧刚性食指弱信号与本次柔顺食指过亮是不同问题；本次新发现不能倒推旧食指偏暗一定由同一原因造成。
实验支持仿真算法和接触形成机制的改进，不能替代实物力—位移、皮肤材料、硬件像素及完整论文误差表的标定。

来源
官方仓库及配置（commit e4f8827407e15625368ef0b9cf5a72c44304e16b）：https://github.com/sharpa-robotics/sharpa-tacmap
论文：https://arxiv.org/pdf/2602.21625
博客：https://www.sharpa.com/blogs/research/tacmap-breaking-the-sim-to-real-deadlock-in-tactile-simulation-with-a-geometric-language
PhysX柔顺接触：https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/docs/RigidBodyDynamics.html#compliant-contacts
'''
    (BASE/'IMPROVEMENT_VALIDATION_REPORT.txt').write_text(text)
    summary=f"完整案例：{'通过' if final['success'] else '未通过'}\n提升 {final['max_lift_mm']:.2f} mm；落点误差 {final['final_xy_error_mm']:.2f} mm\n八例同起终点：{batch['successes']}/8（旧刚性基线8/8）\n\n食指官方原图含跨镊臂射线误接受，全程灰峰241。\n逐目标离线复算后峰155；五指峰值：{replay_peaks}。\n原始图不改写，旁路是场景适配扩展，不是原版官方输出。\n\n新入口为实验版本，任务鲁棒性尚未达到旧基线；保留原默认。\n柔顺280/20、rest=0；重建预持镊和闭合，增加实际夹力门控。\n统一表面与原始深度记录，保留原摩擦、弹簧和关节力矩限幅。\n\n失败试验全部保留；灰度不是压力或实物标定。\n皮肤/壳体精确力归属、非凸目标和硬件标定仍有边界。\n\n详细说明、命令和边界见同目录《IMPROVEMENT_VALIDATION_REPORT.txt》。"
    figures=[BASE/'visualizations/W01_逐目标CPD对照.png',BASE/'visualizations/task_improvement_curves.png',BASE/'visualizations/task_improvement_fixed_grayscale.png',
             BASE/'observation_checks/compliant_v1_full_validation_v2_frame399/diagnostic_comparison.png',BASE/'review/replay8_comparison.png',
             BASE/'single_blade/run_2x2/single_blade_2x2_response.png',BASE/'single_blade/run_2x2/single_blade_2x2_near20mN.png',
             BASE/'review/cold_extended025_tactile_validation.png',BASE/'observation_checks/compliant_v1_full_validation_cooked_geometry.png',BASE/'observation_checks/multi_target_CPD_env3_thumb.png']
    with PdfPages(BASE/'improvement_validation报告.pdf') as pdf:
        fig=plt.figure(figsize=(11.7,8.3));fig.text(.06,.92,'W01 镊子任务：improvement_validation',fontsize=23);fig.text(.06,.84,summary,va='top',fontsize=14,linespacing=1.8);pdf.savefig(fig);fig.savefig(BASE/'visualizations/improvement_validation报告_封面.png',dpi=130);plt.close(fig)
        for path in figures:
            if path.exists():
                fig=plt.figure(figsize=(11.7,8.3));ax=fig.add_axes([.02,.04,.96,.92]);ax.imshow(plt.imread(path));ax.axis('off');fig.text(.03,.015,str(path.relative_to(BASE)),fontsize=7);pdf.savefig(fig);plt.close(fig)
    print(str(BASE/'improvement_validation报告.pdf'))


if __name__=='__main__':main()
