"""Offline audit: native shape reports versus the two tensor contact readers."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'sharpa_w01_tactile_validation/improvement_validation'
OUT = BASE/'review'
OUT.mkdir(exist_ok=True)
font = BASE/'assets/NotoSansSC-Regular.ttf'
font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),
                    'axes.unicode_minus':False, 'font.size':11})
names = ['shape_attribution_smoke', 'shape_attribution_diagnostic']
report = {'status':'NATIVE_SHAPE_ATTRIBUTION_UNAVAILABLE',
          'conclusion':'The tensor readers confirm index distal-phalanx normal contact. The native shape report is empty; neither skin nor shell attribution is established.',
          'units':'get_contact_force_data(dt=physics_dt) returns force in N. Native ContactData.impulse is in N*s and must be divided by the same physics dt for a force comparison. Empty native reports are missing observations, not zero contact force.',
          'runs':{}}
fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor='white')
for i, name in enumerate(names):
    p = BASE/'runs'/name
    ts = np.load(p/'timeseries.npz'); tactile = np.load(p/'tactile.npz')
    native = json.loads((p/'contact_shapes.summary.json').read_text())
    metadata = json.loads((p/'tactile.json').read_text())
    raw = tactile['normal_records']
    n = len(tactile['substep_time_s'])
    raw_vector = np.zeros((n,5,3)); raw_load = np.zeros((n,5))
    ids = raw[:,0].astype(int), raw[:,1].astype(int)
    np.add.at(raw_vector, ids, raw[:,9:12])
    np.add.at(raw_load, ids, np.linalg.norm(raw[:,9:12],axis=1))
    finger = tactile['substep_normal_force_w_N']
    tool = ts['role_normal_force_w'][:,0]
    dt = tactile['substep_dt_s']
    index_bodies = [b for b in metadata['contact_geometry']['bodies'] if b['role']=='index']
    report['runs'][name] = {
        'native_summary':native, 'native_jsonl_bytes':(p/'contact_shapes.jsonl').stat().st_size,
        'physics_steps':n, 'physics_dt_s':float(dt[0]),
        'tensor_normal_contact_records':len(raw),
        'tensor_action_reaction_max_component_error_N':float(np.abs(finger+tool).max()),
        'tensor_raw_vector_sum_max_error_N':float(np.abs(raw_vector-finger).max()),
        'tensor_raw_magnitude_sum_max_error_N':float(np.abs(raw_load-tactile['substep_normal_load_N']).max()),
        'index_mean_normal_load_N':float(raw_load[:,1].mean()),
        'index_max_normal_load_N':float(raw_load[:,1].max()),
        'index_integrated_normal_impulse_magnitude_Ns':float(np.dot(raw_load[:,1],dt)),
        'index_mean_per_step_normal_impulse_magnitude_Ns':float(np.mean(raw_load[:,1]*dt)),
        'index_body_collider_inventory':index_bodies,
        'native_impulse_force_closure':'Unavailable: native report contains no contact data.',
        'skin_vs_shell':'Unknown; both authored shapes share the measured index DP body.',
        'source_sha256':{f:hashlib.sha256((p/f).read_bytes()).hexdigest() for f in
                         ['timeseries.npz','tactile.npz','contact_shapes.summary.json','source/sharpa_contact_shapes.py']}}
    ax = axes[i,0]
    t = tactile['substep_time_s']
    ax.plot(t, np.linalg.norm(finger[:,1],axis=1)*1000, label='指端读者：食指DP合力模长',lw=2)
    ax.plot(t, np.linalg.norm(tool[:,1],axis=1)*1000, '--', label='镊子读者：对应反作用力模长',lw=1.2)
    ax.set(title=name, xlabel='时间 (s)', ylabel='法向合力模长 (mN)'); ax.grid(alpha=.2);ax.legend(fontsize=9)
    ax = axes[i,1]
    ax.plot(t, np.max(np.abs(finger[:,1]+tool[:,1]),axis=-1)*1e9,color='#9b3c57')
    ax.set(title='同一时刻作用/反作用数值闭合',xlabel='时间 (s)',ylabel='最大分量误差 (nN)');ax.grid(alpha=.2)
    ax.text(.04,.94, '原生shape报告：0条，归属不可判定\n未将缺失报告绘制成“0 N接触”',
            va='top',transform=ax.transAxes,fontsize=11,color='#81394b',bbox=dict(facecolor='white',alpha=.9,edgecolor='#dddddd'))
fig.suptitle('合并DP接触力可确认；皮肤/壳体归属仍缺少原生shape证据',fontsize=18)
fig.tight_layout(rect=[0,.02,1,.94])
fig.savefig(OUT/'shape_attribution_tensor_audit.png',dpi=170)
(OUT/'shape_attribution_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({n:{k:v for k,v in s.items() if k in ('index_mean_normal_load_N','index_max_normal_load_N','tensor_action_reaction_max_component_error_N','index_integrated_normal_impulse_magnitude_Ns')} for n,s in report['runs'].items()},indent=2))
