"""Compare recorded task runs without rescaling or synthesizing tactile data."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/rl_scene.py').exists())
BASE = ROOT/'sharpa_w01_tactile_validation/improvement_validation'
font_manager.fontManager.addfont(str(BASE/'assets/NotoSansSC-Regular.ttf'))
plt.rcParams.update({'font.family':'Noto Sans SC','axes.unicode_minus':False,'font.size':11})


def read_run(path, label):
    path = Path(path)
    evaluation = json.loads((path/'evaluation.json').read_text())
    tr = np.load(path/'trajectory.npz')
    tac = np.load(path/'tacmap.npz')
    force = np.load(path/'tactile.npz')
    times = tac['frame_time_s']
    images = tac['deformation_uint8']
    steps = np.rint(times/float(tr['dt'])).astype(int)-1
    assert np.allclose((steps+1)*float(tr['dt']), times)
    assert np.allclose(force['frame_time_s'], times)
    phase = tr['phase'][steps,0]
    ids = np.flatnonzero(phase==5)
    if not len(ids): ids = np.flatnonzero(phase==4)
    if not len(ids): ids = np.arange(len(times))
    index = int(ids[len(ids)//2])
    peak = images.max(axis=(2,3))
    metrics = dict(label=label, path=str(path.resolve()),
        success=bool(evaluation['success_per_env'][0]),
        duration_s=evaluation['duration_s'][0], max_lift_mm=evaluation['max_lift_m'][0]*1000,
        final_xy_error_mm=evaluation['final_xy_error_m'][0]*1000,
        max_phase=evaluation['max_phase'][0], gray_peaks=peak.max(axis=0).tolist(),
        selected_frame=index, selected_time_s=float(times[index]), selected_phase=int(phase[index]),
        selected_gray_peaks=peak[index].tolist(),
        selected_normal_load_N=force['normal_load_N'][index].tolist(),
        selected_normal_vector_mean_magnitude_N=np.linalg.norm(force['normal_force_w_N'][index],axis=-1).tolist(),
        archives_sha256={f:hashlib.sha256((path/f).read_bytes()).hexdigest()
            for f in ['trajectory.npz','tacmap.npz','tactile.npz']})
    carry = np.isin(tr['phase'][:,0], [4,5])
    metrics['face_fraction_all'] = float(np.mean(tr['face'][:,0]))
    metrics['face_fraction_lift_transfer'] = float(np.mean(tr['face'][carry,0])) if np.any(carry) else None
    metrics['minimum_jaw_forces_lift_transfer_N'] = tr['forces'][carry,0].min(axis=0).tolist() if np.any(carry) else None
    if 'diagnostic_raw_cpd_depth_m' in tac.files:
        raw=tac['diagnostic_raw_cpd_depth_m']
        metrics['raw_cpd_peak_um']=raw.max(axis=(0,2,3)).astype(float).tolist()
        metrics['raw_cpd_peak_um']=[v*1e6 for v in metrics['raw_cpd_peak_um']]
    return dict(metrics=metrics,tr=tr,times=times,images=images,peak=peak,force=force,steps=steps,phase=phase)


def main():
    p=argparse.ArgumentParser();p.add_argument('--final-run',required=True);args=p.parse_args()
    runs=[read_run(ROOT/'runs/sharpa_w01_official_tacmap_matched_v2_video','旧任务：刚性接触'),
          read_run(BASE/'runs/soft_original_control','仅改柔顺：旧控制'),
          read_run(args.final_run,'柔顺＋新控制：官方多目标原图')]
    out=BASE/'visualizations';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(4,3,figsize=(16,12),sharex='col',constrained_layout=True)
    for col,r in enumerate(runs):
        m=r['metrics']; tr=r['tr']; time=(np.arange(len(tr['phase']))+1)*float(tr['dt'])
        axes[0,col].set_title(m['label']+'\n'+('任务通过' if m['success'] else '任务未通过'))
        axes[0,col].plot(time,tr['lift'][:,0]*1000,color='#24768d');axes[0,col].axhline(45,color='gray',ls='--',lw=.8)
        axes[0,col].set_ylim(-3,95)
        for i,c,label in [(0,'#d17625','拇指'),(1,'#2176aa','食指')]:
            axes[1,col].plot(r['times'],r['peak'][:,i],color=c,label=label)
            axes[3,col].plot(r['force']['frame_time_s'],r['force']['normal_load_N'][:,i],color=c,label=label)
        axes[1,col].set_ylim(-3,255)
        axes[2,col].plot(time,tr['forces'][:,0,0],label='左镊尖',color='#507755')
        axes[2,col].plot(time,tr['forces'][:,0,1],label='右镊尖',color='#9e6387')
        axes[2,col].axhline(.0604,color='gray',ls='--',lw=.8,label='新夹紧门槛')
        for ax in axes[:,col]: ax.grid(alpha=.2);ax.set_xlim(0,max(28,time[-1]))
        axes[3,col].set_xlabel('仿真时间 (s)')
        axes[1,col].legend(loc='upper right',fontsize=9)
        axes[2,col].legend(loc='upper right',fontsize=9)
    for ax,label in zip(axes[:,0],['方块提升 (mm)','原始 TacMap 峰值 (0–255)','镊尖→方块接触合力 (N)','合并 DP 法向载荷 Σ|Fn| (N)']):ax.set_ylabel(label)
    fig.suptitle('同一搬运起终点：任务结果与触觉观测联合检验\n新方案食指原图高亮含跨镊臂射线误接受；逐目标复算另见对照',fontsize=17)
    fig.savefig(out/'task_improvement_curves.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(12,9),constrained_layout=True)
    for col,r in enumerate(runs):
        m=r['metrics'];i=m['selected_frame']
        for row,finger in enumerate([0,1]):
            ax=axes[row,col];im=ax.imshow(r['images'][i,finger],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            ax.set_title(f"{m['label']}\n{['拇指','食指'][row]}：峰值 {m['selected_gray_peaks'][finger]}；t={m['selected_time_s']:.2f}s；阶段 {m['selected_phase']}",fontsize=11)
    fig.colorbar(im,ax=axes,shrink=.65,ticks=[0,64,128,192,255],label='官方原始 uint8；固定灰度范围')
    fig.suptitle('官方多目标原图对照（不做归一化或增亮）\n新方案食指高亮含已证实的跨镊臂误接受；各列实际几何、受力不同',fontsize=15)
    fig.savefig(out/'task_improvement_fixed_grayscale.png',dpi=170);plt.close(fig)
    report={'runs':[r['metrics'] for r in runs], 'comparison_limits':[
        'Same block start/goal and fixed TacMap quantizer; controller/physical trajectories differ.',
        'Global gray peaks are descriptive, not hardware realism or pressure scores.',
        'The final official multi-target index image includes confirmed outside-origin cross-blade CPD false acceptance; inspect separate per-target replay.',
        'DP normal load is sum of magnitudes, not isolated skin force; scalar and vector means differ.',
        'One case cannot establish randomized grasping success rate.']}
    (out/'task_improvement_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with (out/'task_improvement_metrics.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['run','success','lift_mm','final_error_mm','duration_s','thumb_gray_peak','index_gray_peak'])
        for r in runs:
            m=r['metrics'];w.writerow([m['label'],m['success'],m['max_lift_mm'],m['final_xy_error_mm'],m['duration_s'],*m['gray_peaks'][:2]])
    print(json.dumps([r['metrics'] for r in runs],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
