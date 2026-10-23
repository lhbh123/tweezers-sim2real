#!/usr/bin/env python3
"""Package finalized official TacMap data/evidence without bundling Isaac Sim.

--plan-only is read-only and can run while final evidence is pending. Actual
creation requires successful task, TacMap, PhysX tactile, analytic geometry and
matched-control checks plus a finalized report. Existing ZIPs are never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
SMALL_JSON_LIMIT = 2_000_000


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def evidence(path, estimate=False):
    digest = hashlib.sha256()
    compressor = zlib.compressobj(6, zlib.DEFLATED, -15) if estimate else None
    size, compressed = 0, 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk); size += len(chunk)
            if compressor is not None:
                compressed += len(compressor.compress(chunk))
    if compressor is not None:
        compressed += len(compressor.flush())
    return {'workspace_path': str(path), 'size_bytes': size, 'sha256': digest.hexdigest(),
            **({'estimated_deflated_bytes': compressed} if estimate else {})}


def read(path):
    return json.loads(path.read_text())


def acceptance(run, controls, report_path):
    """Fail closed if a necessary result is missing, stale, or unsuccessful."""
    evaluation = read(run / 'evaluation.json')
    task = read(run / 'validation.json')
    tacmap = read(run / 'tacmap_validation.json')
    tactile = read(run / 'tactile_validation.json')
    geometry = read(run / 'independent_tacmap_task_geometry.json')
    control = read(controls / 'controls_validation.json')
    report = read(report_path)
    episodes = evaluation['episodes']
    assertions = {
        'task_success': episodes > 0 and evaluation['successes'] == episodes,
        'task_independent_claims_consistent': task.get('all_claims_consistent') is True,
        'task_independent_success_count': task.get('reconstructed_successes') == episodes,
        'no_runtime_object_control': task.get('no_runtime_object_control_calls') is True,
        'archived_sources_match': task.get('archived_source_hashes_match') is True,
        'source_geometry_no_table_or_joint_limit_violation':
            task.get('source_mesh_table_audit', {}).get('below_table_link_samples') == 0 and
            task.get('source_mesh_table_audit', {}).get('joint_limit_violations') == 0,
        'tool_geometry_no_table_violation':
            task.get('tool_table_source_box_audit', {}).get('below_table_body_samples') == 0,
        'objects_remain_dynamic': task.get('USD_audit', {}).get('all_objects_dynamic_with_gravity') is True,
        'no_robot_tool_or_block_attachment':
            task.get('USD_audit', {}).get('forbidden_tool_robot_or_block_joints') == [],
        'official_tacmap_valid': tacmap.get('valid') is True,
        'physx_tactile_valid': tactile.get('valid') is True,
        'independent_analytic_geometry_valid': geometry.get('valid') is True,
        'matched_controls_valid': control.get('valid') is True,
        'tacmap_validation_matches_current_npz': tacmap.get('tacmap_npz_sha256') == sha(run / 'tacmap.npz'),
        'geometry_validation_matches_current_npz': geometry.get('tacmap_npz_sha256') == sha(run / 'tacmap.npz'),
        'tactile_validation_matches_current_npz': tactile.get('tactile_npz_sha256') == sha(run / 'tactile.npz'),
        'final_report_refers_to_this_run': run.name in json.dumps(report),
        'final_report_not_pending': str(report.get('status', '')).lower() not in
            ('pending', 'running', 'in_progress', 'draft'),
    }
    failed = [name for name, passed in assertions.items() if not passed]
    if failed:
        raise RuntimeError('Packaging acceptance checks failed: ' + ', '.join(failed))
    return assertions


def collect(args):
    included, excluded, missing = {}, {}, []
    run, controls, baseline, maps = map(resolve, (args.run, args.controls, args.baseline, args.maps))

    def omit(path, reason):
        path = resolve(path)
        if path.is_file():
            excluded[path] = reason

    def add(path, required=False):
        path = resolve(path)
        if not path.is_file():
            if required:
                missing.append(str(path))
            return
        if not path.is_relative_to(ROOT):
            raise ValueError('Only workspace files may be archived: ' + str(path))
        included[path.relative_to(ROOT).as_posix()] = path
        excluded.pop(path, None)

    def small_reports(folder):
        for path in sorted(folder.glob('*.json')):
            if path.stat().st_size <= SMALL_JSON_LIMIT:
                add(path)
            else:
                omit(path, 'Large JSON; retained in workspace')

    for name in ('tacmap.npz', 'tactile.npz', 'trajectory.npz', 'evaluation.json',
                 'validation.json', 'tacmap_validation.json', 'tactile_validation.json',
                 'independent_tacmap_task_geometry.json', 'task.json', 'launch.json'):
        add(run / name, required=True)
    small_reports(run)
    for path in sorted((run / 'source').rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            add(path)
    for path in sorted(run.iterdir()) if run.exists() else []:
        if path.is_file() and (path.suffix in ('.mp4', '.usda', '.usd', '.jsonl') or
                              (path.suffix == '.npz' and path.name not in ('tacmap.npz', 'tactile.npz', 'trajectory.npz'))):
            omit(path, 'Large native scene/video/substep trace; data-evidence bundle only')

    add(controls / 'controls.npz', required=True)
    add(controls / 'controls_validation.json', required=True)
    small_reports(controls)
    add(controls / 'fixed_geometry_controls.usda')
    small_reports(baseline)
    if (baseline / 'tacmap.npz').is_file():
        add(baseline / 'tacmap.npz')  # Optional; budget selection below may omit it.
    for name in ('tactile.npz', 'trajectory.npz', 'grasp_substeps.jsonl', 'final_scene.usda'):
        omit(baseline / name, 'Original-map baseline: compact reports and optionally TacMap NPZ included')
    for path in baseline.glob('*.mp4'):
        omit(path, 'Baseline native video omitted')
    original_controls = ROOT / 'runs/sharpa_official_tacmap_controls_v4'
    small_reports(original_controls)
    omit(original_controls / 'controls.npz', 'Original-map control raw arrays remain in workspace; matched controls included')

    for path in sorted(maps.iterdir()) if maps.exists() else []:
        if not path.is_file():
            continue
        if path.suffix == '.npz':
            omit(path, 'Large per-taxel correspondence arrays; regenerable with included geometry-only generator')
        else:
            add(path)
    add(maps / 'manifest.json', required=True)
    for suffix, tail in (('TH', ''), ('4F', '_origin')):
        for kind in ('point', 'normal'):
            add(maps / f'tactileSensor_map_{suffix}_{kind}{tail}.npy', required=True)

    official = ROOT / 'assets/sharpa_tactile'
    for relative in ('__init__.py', 'tacmap_sensor/__init__.py', 'tacmap_sensor/sharpa_tacmap_vbts.py',
                     'tacmap_sensor/sharpa_tacmap_cfg.py', 'torch_jit_utils.py', 'env.py', 'env_cfg.py',
                     'README.md', 'LICENSE', 'NOTICE'):
        add(official / relative, required=True)
    for path in sorted((official / 'assets/tactilesensor_map').glob('*.npy')):
        omit(path, 'Original published surface maps obtainable from recorded pinned official repository')
    for name in ('LICENSE.txt', 'NOTICE.txt'):
        add(ROOT / 'assets/sharpa_wave/official' / name, required=True)
    for path in sorted((ROOT / 'src/sharpa_tacmap_vendor').rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            add(path)
    for path in sorted((ROOT / 'src').glob('*.py')):
        add(path)
    for pattern in ('*tacmap*.py', '*tacmap*.sh', 'run_pick_place.py', 'run_sharpa_w01.sh', 'build_sharpa_robot.py'):
        for path in sorted((ROOT / 'scripts').glob(pattern)):
            add(path)
    for pattern in ('*tacmap*.py', 'validate_pick_place.py', 'validate_sharpa_tactile.py'):
        for path in sorted((ROOT / 'analysis').glob(pattern)):
            add(path)
    for path in sorted((ROOT / 'analysis/sharpa_tacmap').glob('*.py')):
        add(path)
    for path in sorted((ROOT / 'analysis/sharpa_tacmap').glob('*.json')):
        if path.name.startswith('package_plan_'):
            continue
        if path.stat().st_size <= SMALL_JSON_LIMIT:
            add(path)
        else:
            omit(path, 'Large diagnostic JSON; compact diagnostic summaries included')
    for path in sorted((ROOT / 'analysis/sharpa_tacmap').glob('*.npz')):
        omit(path, 'Intermediate extracted geometry retained in workspace')
    add(ROOT / 'analysis/sharpa_research/sharpa-tacmap_tree.json')
    for path in sorted(ROOT.glob('README*.md')):
        add(path)

    launch = read(run / 'launch.json') if (run / 'launch.json').is_file() else {}
    for key in ('pose', 'snapshot', 'camera_config', 'cases'):
        value = launch.get('arguments', {}).get(key)
        if value:
            add(value, required=True)
    # Source assets are purposely excluded, with exact paths/hashes retained.
    robot_hashes = run / 'robot_asset_sha256.json'
    if robot_hashes.is_file():
        for path in read(robot_hashes):
            omit(path, 'Robot asset excluded; requires the original project and Isaac installation')
            if Path(path).suffix == '.urdf' and Path(path).is_file():
                urdf = ET.parse(path).getroot()
                for mesh in urdf.findall('.//mesh'):
                    name = mesh.get('filename', '')
                    if name and '://' not in name:
                        omit(Path(path).parent / name, 'Referenced source robot mesh excluded')
    add(args.report, required=True)
    for path in args.extra:
        add(path, required=True)
    for path in args.omit:
        omit(path, 'Final media artifact delivered separately')
    omit(ROOT / 'assets/fonts/NotoSansCJKsc-Regular.otf',
         'Large composition font omitted; obtain from the existing project assets')
    return included, excluded, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('runs/sharpa_w01_official_tacmap_matched_v2_video'))
    parser.add_argument('--controls', type=Path, default=Path('runs/sharpa_official_tacmap_controls_matched_v1'))
    parser.add_argument('--baseline', type=Path, default=Path('runs/sharpa_w01_official_tacmap_v1_video'))
    parser.add_argument('--maps', type=Path, default=Path('assets/sharpa_tacmap_w01_matched_v2'))
    parser.add_argument('--report', type=Path, default=Path('reports/sharpa_w01_official_tacmap_final.json'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--target-bytes', type=int, default=28_000_000)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--extra', type=Path, action='append', default=[])
    parser.add_argument('--omit', type=Path, action='append', default=[])
    args = parser.parse_args()
    run, controls, report_path = map(resolve, (args.run, args.controls, args.report))
    files, omitted, missing = collect(args)
    if not args.plan_only:
        if missing:
            raise RuntimeError('Required files pending: ' + ', '.join(missing))
        if args.output is None:
            parser.error('--output is required unless --plan-only')
        output = resolve(args.output)
        if output.exists():
            raise FileExistsError(output)
        if output.with_suffix('.verification.json').exists():
            raise FileExistsError(output.with_suffix('.verification.json'))
        gates = acceptance(run, controls, report_path)
    else:
        gates = {'not_checked': 'Read-only plan; no ZIP will be created'}
    included = {name: evidence(path, estimate=True) for name, path in sorted(files.items())}
    estimate = sum(row['estimated_deflated_bytes'] for row in included.values()) + 150_000
    optional = resolve(args.baseline) / 'tacmap.npz'
    optional_name = optional.relative_to(ROOT).as_posix()
    if estimate > args.target_bytes and optional_name in included:
        estimate -= included[optional_name]['estimated_deflated_bytes']
        included.pop(optional_name); files.pop(optional_name)
        omitted[optional] = 'Optional baseline TacMap NPZ omitted to meet compact evidence-bundle target'
    plan = {'plan_only': args.plan_only, 'file_count': len(included), 'estimated_zip_bytes': estimate,
            'target_bytes': args.target_bytes, 'target_exceeded': estimate > args.target_bytes,
            'missing_required_files': missing, 'optional_baseline_npz_included': optional_name in included,
            'largest_included': sorted([{'archive_path': name, **row} for name, row in included.items()],
                                       key=lambda row: row['estimated_deflated_bytes'], reverse=True)[:12]}
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2)); return
    excluded = [{**evidence(path), 'reason': reason} for path, reason in sorted(omitted.items(), key=lambda item: str(item[0]))]
    readme = (
        'Official Sharpa TacMap — compact data and evidence bundle\n\n'
        'This is a data/evidence archive, not a self-contained Isaac Sim installation.\n'
        'Workspace-relative paths are retained. Read reports/sharpa_w01_official_tacmap_final.json\n'
        'for final outcomes, methods and limitations. Root project READMEs describe reproduction.\n\n'
        'Main tacmap.npz: official uint8 geometry-penetration codes after the unchanged nonlinear\n'
        'quantizer and Gaussian blur. The current-W01 input maps are derived and explicitly labeled;\n'
        'they are not untouched published maps or hardware UV calibration. No force-to-depth\n'
        'conversion or arbitrary outward map offset is used. Codes are not N, pressure, or FEM.\n'
        'tactile.npz independently records PhysX normal/friction forces and original contact data.\n'
        'trajectory.npz preserves task states. Load NPZ with numpy.load(..., allow_pickle=False).\n\n'
        'manifest.json inventories every payload file by source path, byte count and SHA256.\n'
        'Excluded large scenes, native videos, full substep JSONL, robot assets and correspondence\n'
        'arrays are listed with workspace paths/hashes; obtain these separately or regenerate\n'
        'with the included scripts and pinned dependencies. The external .verification.json\n'
        'receipt hashes the ZIP and manifest (the manifest cannot hash itself).\n'
    ).encode('utf-8')
    included['BUNDLE_README.txt'] = {'workspace_path': None, 'size_bytes': len(readme),
                                   'sha256': hashlib.sha256(readme).hexdigest(), 'generated': True}
    manifest = {'schema_version': 1, 'bundle_kind': 'data_and_evidence_not_isaac_installation',
                'main_run': str(run), 'final_report_sha256': sha(report_path), 'acceptance': gates,
                'included': included, 'excluded': excluded, 'size_plan': plan,
                'manifest_self_hash': 'Stored in external verification receipt'}
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            digest = hashlib.sha256(); size = 0
            with path.open('rb') as stream, archive.open(name, 'w') as dest:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    dest.write(chunk); digest.update(chunk); size += len(chunk)
            if digest.hexdigest() != included[name]['sha256'] or size != included[name]['size_bytes']:
                raise RuntimeError('Input changed while packaging: ' + str(path))
        archive.writestr('BUNDLE_README.txt', readme)
        archive.writestr('manifest.json', manifest_bytes)
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError('ZIP CRC failed: ' + bad)
        if set(archive.namelist()) != set(included) | {'manifest.json'}:
            raise RuntimeError('ZIP membership differs from manifest')
        for name, row in included.items():
            digest = hashlib.sha256(); size = 0
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk); size += len(chunk)
            if digest.hexdigest() != row['sha256'] or size != row['size_bytes']:
                raise RuntimeError('Archived file SHA256/size failed: ' + name)
    result = {'file': str(output), 'bytes': output.stat().st_size, 'sha256': sha(output),
              'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
              'payload_files_verified': len(included), 'excluded_files': len(excluded),
              'zip_crc_valid': True, 'all_payload_sha256_and_sizes_valid': True,
              'target_bytes': args.target_bytes, 'target_exceeded': output.stat().st_size > args.target_bytes}
    receipt = output.with_suffix('.verification.json')
    with receipt.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
