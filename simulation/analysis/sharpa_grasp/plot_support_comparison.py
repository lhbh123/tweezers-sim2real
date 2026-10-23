"""Plot a recorded loose-grip pair; never run or alter a physical experiment.

python3 analysis/sharpa_grasp/plot_support_comparison.py RUN --pair 0 1 \
  --out RUN/support_comparison_01.png --note '开发图：B 强松握，零承托、失握'

Requires the independent natural_support_audit.json and original probe NPZ.
All force and state traces retain their saved physics-step sampling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

from audit_natural_support import load_probe, read_json, sha256

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--pair', type=int, nargs=2, default=[0, 1], metavar=('IN_PLACE', 'RETRACTED'))
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--note', default='')
    parser.add_argument('--title', default='真实松握对照：中指在位与收回')
    parser.add_argument('--pdf', action='store_true')
    args = parser.parse_args()
    run = args.run
    audit_path = args.audit or run/'natural_support_audit.json'
    audit = read_json(audit_path)
    if Path(audit['run']).resolve() != run.resolve():
        raise ValueError('Audit belongs to another run')
    original = run/'timeseries.npz'
    hashes = [value for name,value in audit['source_sha256'].items() if Path(name).name == original.name]
    if hashes != [sha256(original)]:
        raise ValueError('Original probe NPZ does not match audited input hash')
    rows, pose, masses, centres, source = load_probe(run)
    lookup = {r['environment']:r for r in rows}
    audit_lookup = {r['environment']:r for r in audit['rows']}
    selected = [lookup[i] for i in args.pair]
    selected_audit = [audit_lookup[i] for i in args.pair]
    if selected_audit[0]['baseline_window_s'] != selected_audit[1]['baseline_window_s']:
        raise ValueError('Pair uses different baseline windows')
    report = read_json(run/'report.json')
    d = np.load(original)
    assignment = report['face_pinch_assignment']
    font_path = ROOT/'assets/fonts/NotoSansCJKsc-Regular.otf'
    if not font_path.is_file():
        raise FileNotFoundError(f'Chinese plot font unavailable: {font_path}')
    font = FontProperties(fname=str(font_path))
    plt.rcParams.update({'axes.unicode_minus':False, 'axes.spines.top':False,
                         'axes.spines.right':False, 'font.size':11,
                         'figure.facecolor':'white', 'axes.facecolor':'white',
                         'svg.fonttype':'none'})
    colors = ['#c46b18', '#2273ad']
    labels = ['中指在位', '中指收回']
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(13.2, 9.4),
                             gridspec_kw={'height_ratios':[1.1, 1.1, .92]})
    fig.subplots_adjust(left=.145, right=.97, top=.84, bottom=.13, hspace=.20)
    fig.suptitle(args.title, x=.145, y=.975, ha='left', fontproperties=font, fontsize=22)
    if args.note:
        fig.text(.145, .932, args.note, fontproperties=font, fontsize=13, color='#9b352a')
    summaries = []
    dt = source['physics_dt_s']
    max_force = 0.
    for n, (row, independent) in enumerate(zip(selected, selected_audit)):
        env = row['environment']
        t = row['time']
        normal, friction, net = row['normal_z'], row['friction_z'], row['net_z']
        color = colors[n]
        axes[0].plot(t, normal, color=color, linestyle=':', linewidth=1., alpha=.65)
        axes[0].plot(t, friction, color=color, linestyle='--', linewidth=1., alpha=.65)
        axes[0].plot(t, net, color=color, linewidth=1.8, label=labels[n])
        max_force = max(max_force, float(np.max(np.abs(np.r_[normal, friction, net]))))
        drop = 1000.*(independent['baseline_COM_world_z_m']-row['com'][:, 2])
        axes[1].plot(t, drop, color=color, linewidth=1.8)
        pairs = [(['left','right'].index(assignment[f]), ['thumb','index'].index(f))
                 for f in ['thumb','index']]
        inward = np.stack([d['broad_face_inward_force_N'][:, env, j, role] for j,role in pairs], axis=-1)
        face = (inward > audit['thresholds']['contact_and_net_upward_N']).all(-1)
        held = (row['grip'] < audit['thresholds']['retained_grip_error_m']) & (
            row['rotation'] < audit['thresholds']['retained_rotation_sin_half_squared'])
        for status, y in [(face, 3.-2*n), (held, 2.-2*n)]:
            axes[2].fill_between(t, y-.25, y+.25, color='#f7dddd', linewidth=0)
            axes[2].fill_between(t, y-.25, y+.25, where=status,
                                  color=color, alpha=.9, linewidth=0, step='post')
        window = independent['windows'].get('loose', independent['windows']['whole_run'])
        summaries.append(f'{labels[n]}：中指支撑 {window["qualifying_under_support_duration_s"]:.3f} s'
                         f'，最大 COM 下降 {window["COM_drop_from_baseline_max_mm"]:.2f} mm')
    for ax in axes:
        ax.grid(axis='y', color='#e5e9ed', linewidth=.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10)
        ax.set_xlim(0., max(r['time'][-1] for r in selected))
        for independent in selected_audit[:1]:
            loose = independent['windows'].get('loose')
            if loose and loose['samples']:
                start = loose['first_sample_end_s']-dt
                end = loose['last_sample_end_s']
                ax.axvspan(start, end, color='#d9e1e8', alpha=.27, linewidth=0, zorder=0)
                ax.axvline(start, color='#8b98a5', linestyle=':', linewidth=.8)
                ax.axvline(end, color='#8b98a5', linestyle=':', linewidth=.8)
    axes[0].axhline(0., color='#808890', linewidth=.65, zorder=1)
    axes[1].axhline(0., color='#808890', linewidth=.65, zorder=1)
    axes[0].set_ylabel('中指施于工具的向上力 / N', fontproperties=font, fontsize=12)
    axes[1].set_ylabel('相对基线的 COM 下降 / mm', fontproperties=font, fontsize=12)
    axes[2].set_yticks([3., 2., 1., 0.], ['在位：双宽面', '在位：仍持握', '收回：双宽面', '收回：仍持握'],
                      fontproperties=font, fontsize=11)
    axes[2].set_ylim(-.5, 3.6)
    axes[2].set_xlabel('仿真时间 / s', fontproperties=font, fontsize=12)
    axes[2].grid(False)
    legend = [Line2D([0],[0],color=colors[n],linewidth=2.,label=labels[n]) for n in range(2)]
    legend += [Line2D([0],[0],color='#626b73',linewidth=1.6,linestyle=style,label=label)
               for style,label in [('-', '净向上力 = 法向 + 摩擦'), (':','法向分量'), ('--','摩擦分量')]]
    axes[0].legend(handles=legend, prop=font, loc='upper left', ncol=3, frameon=False, fontsize=10)
    if max_force < 1e-12:
        axes[0].set_ylim(-.01, .025)
        axes[0].text(.98, .18, '两组中指受力均严格为 0 N', transform=axes[0].transAxes,
                     ha='right', fontproperties=font, fontsize=12, color='#92372c')
    axes[2].text(.995, 1.02, '彩色：满足条件    浅红：未满足', ha='right',
                 transform=axes[2].transAxes, fontproperties=font, fontsize=10)
    fig.text(.145, .884, '  |  '.join(summaries), fontproperties=font, fontsize=11)
    baseline = selected_audit[0]['baseline_window_s']
    footer = (f'来源：{run.name}，环境 {args.pair[0]} / {args.pair[1]}；'
              f'原始采样 {1./dt:.0f} Hz，未平滑；COM 基线 {baseline[0]:g}–{baseline[1]:g} s。\n'
              '灰底为松握观测窗口；中指下方接触与向上净力共同计入支撑。夹持状态不是夹放任务成功判定。')
    fig.text(.145, .04, footer, fontproperties=font, fontsize=9.5, color='#555f68', linespacing=1.5)
    out = args.out or run/f'support_comparison_{args.pair[0]}{args.pair[1]}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor='white')
    if args.pdf:
        fig.savefig(out.with_suffix('.pdf'), facecolor='white')
    plt.close(fig)
    manifest = {'run':str(run), 'environment_order':args.pair,
                'interpretation':'Middle in place / middle retracted; measured probe records only.',
                'note':args.note, 'time_smoothing':False, 'physics_dt_s':dt,
                'audit':str(audit_path), 'audit_sha256':sha256(audit_path),
                'probe_npz_sha256':sha256(original), 'script_sha256':sha256(Path(__file__)),
                'image':str(out), 'image_sha256':sha256(out), 'summary_labels':summaries}
    out.with_suffix('.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
