"""Read-only analysis of the four bounded-force contact fixtures."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[3]
p = argparse.ArgumentParser()
p.add_argument('--run', type=Path, default=ROOT/'sharpa_w01_tactile_validation/improvement_validation/single_blade/run_2x2')
a = p.parse_args()
font = ROOT/'sharpa_w01_tactile_validation/improvement_validation/assets/NotoSansSC-Regular.ttf'
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.unicode_minus': False})
names = ['rigid_rest0', 'rigid_restneg', 'soft_rest0', 'soft_restneg']
labels = ['刚性 / rest = 0', '刚性 / rest = −0.1 mm',
          'K=280, D=20 / rest = 0', 'K=280, D=20 / rest = −0.1 mm']
colors = ['#3478a7', '#71afc1', '#d27728', '#b33b55']
rows = []
for name in names:
    folder = a.run/name
    d = np.load(folder/'trajectory.npz')
    m = json.loads((folder/'metadata.json').read_text())
    s = json.loads((folder/'summary.json').read_text())
    n = len(d['time_s'])
    depth = d['raw_CPD_m'].reshape(n, -1).max(1)*1e6
    gray = d['gray_u8'].reshape(n, -1).max(1)
    force = np.linalg.norm(d['force_blade_w'], axis=-1)[:, 1]
    # The original saved displacement is sampled before physics; compute the
    # actual displacement of the same post-physics pose as tactile/force data.
    travel = -(d['blade_pose_w'][:, :3]-np.array(m['blade_initial_center_w'])) @ np.array(m['axis_w'])*1e3
    k = int(depth.argmax())
    rows.append(dict(name=name, data=d, metadata=m, summary=s,
                     depth=depth, gray=gray, force=force, travel=travel, peak=k))

metrics = {'definition': 'Four independent bounded-force single-axis dynamic blade coupon fixtures. Not a free tweezers grasp.',
           'actual_travel_definition': 'Post-physics blade world position projected onto the initial surface outward normal; positive toward skin. Saved actual_rail_displacement_m is pre-physics control feedback.',
           'scope': 'Tests local normal contact only; gravity disabled on hand/blade, rail fixes other five degrees of freedom. Cannot establish grasp/slip performance.',
           'case_metrics': {}}
for r in rows:
    d, m, s = r['data'], r['metadata'], r['summary']
    k = r['peak']
    material_ok = True
    for b in m['material_readback']:
        attrs, c = b['material_attributes'], b['collider_attributes']
        blade = b['collider'].endswith('/Blade/Mesh')
        expected_mu = .5 if blade else 1.4
        expected_k = 0. if blade else m['case_parameters']['skin_stiffness']
        expected_d = 0. if blade else m['case_parameters']['skin_damping']
        expected_rest = 0. if blade else m['case_parameters']['skin_rest_m']
        checks = [np.isclose(attrs['physics:staticFriction'], expected_mu),
                  np.isclose(attrs['physics:dynamicFriction'], expected_mu),
                  attrs['physxMaterial:frictionCombineMode'] == 'max',
                  np.isclose(attrs['physxMaterial:compliantContactStiffness'], expected_k),
                  np.isclose(attrs['physxMaterial:compliantContactDamping'], expected_d),
                  np.isclose(c['physxCollision:contactOffset'], .0002),
                  np.isclose(c['physxCollision:restOffset'], expected_rest)]
        material_ok &= bool(all(checks))
    actual_limits = dict(zip(m['joint_names'], m['actual_joint_effort_limits_Nm']))
    joint_limit_ok = all(np.isclose(actual_limits[n], v) for n, v in m['joint_pd']['effort_limits_Nm'].items())
    metrics['case_metrics'][r['name']] = {
        **s, 'material_readback_matches_case': material_ok,
        'joint_effort_readback_matches_URDF': bool(joint_limit_ok),
        'max_actual_poststep_travel_mm': float(r['travel'].max()),
        'depth_peak_frame': k, 'depth_peak_time_s': float(d['time_s'][k]),
        'depth_peak_gray': int(r['gray'][k]), 'depth_peak_force_N': float(r['force'][k]),
        'depth_peak_nonzero_depth_pixels': int(np.count_nonzero(d['raw_CPD_m'][k])),
        'depth_peak_nonzero_gray_pixels': int(np.count_nonzero(d['gray_u8'][k])),
        'trajectory_sha256': hashlib.sha256((a.run/r['name']/'trajectory.npz').read_bytes()).hexdigest()
    }
    approach_ids = np.flatnonzero(d['phase'] == 'approach')
    equal_id = int(approach_ids[np.argmin(np.abs(r['force'][approach_ids]-.020))])
    r['equal_force_frame'] = equal_id
    metrics['case_metrics'][r['name']]['nearest_20mN_approach_sample'] = {
        'frame': equal_id, 'time_s': float(d['time_s'][equal_id]),
        'actual_force_N': float(r['force'][equal_id]),
        'CPD_um': float(r['depth'][equal_id]), 'max_gray': int(r['gray'][equal_id]),
        'actual_poststep_travel_mm': float(r['travel'][equal_id])}

fig = plt.figure(figsize=(16, 11), facecolor='white')
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.15], hspace=.5, wspace=.28)
for j, r in enumerate(rows):
    d, k = r['data'], r['peak']
    ax = fig.add_subplot(gs[0, j])
    im = ax.imshow(d['gray_u8'][k], cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'{labels[j]}\n峰值 CPD {r["depth"][k]:.1f} µm / 灰度 {r["gray"][k]}', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    if j == 0: ax.set_ylabel('官方灰度，统一 0–255')
    ax = fig.add_subplot(gs[1, j])
    imd = ax.imshow(d['raw_CPD_m'][k]*1e6, cmap='magma', vmin=0, vmax=250)
    ax.set_title(f't = {d["time_s"][k]:.2f} s / 力 {r["force"][k]*1000:.1f} mN', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    if j == 0: ax.set_ylabel('原始 CPD，统一 0–250 µm')
axf = fig.add_subplot(gs[2, :2]); axd = fig.add_subplot(gs[2, 2:])
for r, label, color in zip(rows, labels, colors):
    t = r['data']['time_s']
    axf.plot(t, r['force']*1000, color=color, label=label, lw=1.8)
    axd.plot(t, r['depth'], color=color, label=label, lw=1.8)
axf.set(xlabel='仿真时间 (s)', ylabel='食指–镊臂接触力 (mN)', title='接近、接触、释放全过程')
axd.set(xlabel='仿真时间 (s)', ylabel='全图最大原始 CPD (µm)', title='同一有界驱动力控制轨迹；真实接触位姿可不同')
for ax in (axf, axd): ax.grid(alpha=.2); ax.legend(fontsize=9)
fig.suptitle('1 mm 镊臂局部接触：刚性/柔顺 × 皮肤 rest offset', fontsize=21, y=.985)
fig.text(.5, .025, '当前 W01 + matched_v2；皮肤碰撞使用原任务凸分解；真实关节额定力矩；皮肤/镊臂摩擦 1.4/0.5，max combine。\n'
         '动态镊臂受单轴导轨约束，驱动力上限 0.08 N；无运行期位姿重写；此图检验局部法向接触，不代表自由持镊或滑动测试。',
         ha='center', fontsize=10, color='#425164')
fig.subplots_adjust(left=.06, right=.97, top=.91, bottom=.13)
fig.savefig(a.run/'single_blade_2x2_contact.png', dpi=170)
plt.close(fig)

fig, ax = plt.subplots(2, 2, figsize=(13, 9), facecolor='white')
for r, label, color in zip(rows, labels, colors):
    d = r['data']; active = d['phase'] == 'approach'
    ax[0,0].plot(d['time_s'], -d['target_displacement_m']*1000, c=color, ls='--', alpha=.5)
    ax[0,0].plot(d['time_s'], r['travel'], c=color, label=label)
    ax[0,1].plot(r['travel'][active], r['force'][active]*1000, c=color, label=label)
    ax[1,0].plot(r['depth'][active], r['force'][active]*1000, c=color, label=label)
    ax[1,1].plot(d['time_s'], r['gray'], c=color, label=label)
ax[0,0].set(xlabel='仿真时间 (s)', ylabel='向皮肤推进量 (mm)', title='实线：帧后真实位移；虚线：有界驱动目标')
ax[0,1].set(xlabel='帧后真实推进量 (mm)', ylabel='接触力 (mN)', title='接近阶段：真实位移与力')
ax[1,0].set(xlabel='最大原始 CPD (µm)', ylabel='接触力 (mN)', title='接近阶段：CPD 与接触力（并非压力标定）')
ax[1,1].set(xlabel='仿真时间 (s)', ylabel='最大灰度 (0–255)', title='灰度量化后的响应')
for x in ax.ravel(): x.grid(alpha=.2); x.legend(fontsize=9)
fig.suptitle('单镊臂诊断：实际位移、法向力与几何 CPD', fontsize=19)
fig.tight_layout(rect=[0, .015, 1, .95])
fig.savefig(a.run/'single_blade_2x2_response.png', dpi=170)
plt.close(fig)

fig, axs = plt.subplots(2, 4, figsize=(15, 8), facecolor='white')
for j, r in enumerate(rows):
    d, k = r['data'], r['equal_force_frame']
    axs[0,j].imshow(d['gray_u8'][k], cmap='gray', vmin=0, vmax=255)
    axs[0,j].set_title(f'{labels[j]}\n实际力 {r["force"][k]*1000:.2f} mN / 灰度 {r["gray"][k]}', fontsize=10)
    axs[1,j].imshow(d['raw_CPD_m'][k]*1e6, cmap='magma', vmin=0, vmax=250)
    axs[1,j].set_title(f'最大 CPD {r["depth"][k]:.1f} µm', fontsize=10)
    for x in axs[:,j]: x.set_xticks([]); x.set_yticks([])
axs[0,0].set_ylabel('官方灰度，统一 0–255')
axs[1,0].set_ylabel('原始 CPD，统一 0–250 µm')
fig.suptitle('约 20 mN 相近接触力下，皮肤 rest offset 与柔顺参数改变 CPD', fontsize=18, y=.98)
fig.text(.5, .035, '选各组接近阶段最接近 20 mN 的真实采样帧；实际力标于图中，未插值、未调灰度增益。\n'
         '该结果展示接触配置与几何压入量的关系，不是传感器压力标定，也不验证自由持镊任务。',
         ha='center', fontsize=10, color='#425164')
fig.subplots_adjust(left=.055, right=.98, top=.88, bottom=.15, wspace=.22, hspace=.22)
fig.savefig(a.run/'single_blade_2x2_near20mN.png', dpi=170)
plt.close(fig)
(a.run/'analysis.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
print(json.dumps(metrics, ensure_ascii=False, indent=2))
