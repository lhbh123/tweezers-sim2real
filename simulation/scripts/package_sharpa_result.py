"""Package measured tactile arrays, experiment evidence and upstream notices."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run', type=Path)
parser.add_argument('--report', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--extra', type=Path, action='append', default=[],
                    help='Additional experiment evidence file to include, repeatable')
args = parser.parse_args()
run = args.run.resolve()
evaluation = json.loads((run / 'evaluation.json').read_text())
report = json.loads(args.report.read_text())
files = {name: run / name for name in [
    'tactile.npz', 'tactile.json', 'trajectory.npz', 'evaluation.json', 'task.json',
    'validation.json', 'tactile_validation.json', 'source_mesh_trajectory_audit.json', 'launch.json']}
for name in ['independent_release_feedback_audit.json', 'independent_gate_audit.json',
             'independent_release_middle_audit.json', 'natural_support_audit.json',
             'natural_posture_geometry.json', 'contact_geometry_review.json',
             'source_overlap_depth.json', 'actual_middle_source_mesh_audit.json',
             'middle_contact_source_audit.json', 'middle_contact_classification.json',
             'unmatched_middle_source_detail.json', 'independent_process_audit.json']:
    if (run / name).exists():
        files[name] = run / name
files['final_report.json'] = args.report.resolve()
for extra in args.extra:
    path = extra.resolve()
    name = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
    files['extra/' + name] = path
for label in ['randomized_evaluation', 'regression_on_previous_eight_cases', 'development_baseline']:
    if label not in report:
        continue
    evidence_run = Path(report[label]['run'])
    for name in ['evaluation.json', 'validation.json', 'launch.json', 'trajectory.npz',
                 'independent_gate_audit.json', 'independent_release_feedback_audit.json',
                 'independent_release_middle_audit.json', 'independent_process_audit.json',
                 'independent_failure_diagnosis.json']:
        if label == 'development_baseline' and name == 'trajectory.npz':
            continue  # Historical raw traces remain in the project; include current test traces here.
        if (evidence_run / name).exists():
            files['experiments/' + label + '/' + name] = evidence_run / name
launch = json.loads((run / 'launch.json').read_text())
for argument in ['pose', 'snapshot', 'camera_config', 'cases']:
    value = launch['arguments'].get(argument)
    if not value:
        continue
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    archived = run / 'source' / path.name
    files['configs/' + path.name] = archived if archived.is_file() else path
for source, names in [
    ('assets/sharpa_wave/official', ['LICENSE.txt', 'NOTICE.txt']),
    ('assets/sharpa_tactile', ['LICENSE', 'NOTICE'])]:
    for name in names:
        files['licenses/' + Path(source).parent.name + '_' + Path(source).name + '/' + name] = ROOT / source / name
manifest = {name: {'source': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in files.items()}
readme = f'''Sharpa Wave W01 仿真触觉数据

本包对应独立录像回合；成功 {evaluation['successes']}/{evaluation['episodes']}，
最高抬升 {1000 * evaluation['max_lift_m'][0]:.3f} mm，最终落点误差 {1000 * evaluation['final_xy_error_m'][0]:.3f} mm。
完整随机/回归测试与控制参数见 final_report.json，不能只凭此单例推断所有位置均成功。

读取：
import numpy as np
z = np.load('tactile.npz', allow_pickle=False)
print(z.files)
print(z['finger_names'])

480 Hz 原始物理步：substep_time_s、substep_normal_load_N、substep_friction_load_N。
30 fps 同步力图：frame_time_s、normal_map_N、friction_map_N。
原始接触矢量/位置：normal_records、friction_records；列定义在 raw_columns。
法向/切向图外力、表面采样位置、有效掩码和记录元数据均保留。
指序：拇指、食指、中指、无名指、小指；无接触时为真实零值。

力图单位是 N/bin，不是压力或实机视触觉图像；旧版官方表面地图与新版源网格有局部差异。
3 mm 为可视化投影匹配容差，不是传感器精度。当前只读取力和摩擦锚点，不含完整六轴扭矩。
本实验从预持镊初态开始，以关节驱动和物理接触完成任务，无运行时工具附着或物体位姿搬运。
W01使用20–85度工作角域；未训练W01 PPO，摩擦及执行器没有实机标定。
源网格审查覆盖保存的控制状态，不等同所有物理子步或连续碰撞检测。

manifest.json 提供逐文件校验；licenses/ 保留上游 W01 与 TacMap 的许可证及NOTICE。
自然握姿说明见 README_SHARPA_W01_NATURAL.md；官方资产与历史版本见 README_SHARPA_W01.md。
复现入口为 scripts/run_sharpa_w01.sh，重放时必须使用本包中的匹配姿态和快照。
几何审查的交叠计数与深度应一起读取，不能将接触位置的源网格交叠误称为全程零穿入。
'''
args.output.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for name, path in files.items():
        archive.write(path, name)
    archive.writestr('README.txt', readme)
    archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
with zipfile.ZipFile(args.output) as archive:
    assert archive.testzip() is None
print(json.dumps({'file': str(args.output), 'bytes': args.output.stat().st_size,
                  'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}))
