"""Package reviewable changes and evidence; large raw runs stay in the workspace."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'src/rl_scene.py').is_file())
BASE = ROOT / 'sharpa_w01_tactile_validation/improvement_validation'


def main():
    paths = set()
    for name in ['IMPROVEMENT_VALIDATION_REPORT.txt', 'improvement_validation报告.pdf', 'final_verification_summary.json']:
        paths.add(BASE / name)
    for name in ['README.md', 'README_SHARPA_W01.md', 'README_SHARPA_W01_NATURAL.md',
                 'scripts/run_sharpa_w01.sh', 'scripts/run_sharpa_w01_compliant.sh',
                 'scripts/run_pick_place.py', 'scripts/compose_sharpa_tacmap_video.py',
                 'configs/sharpa_w01_tacmap_compliant_v1.json',
                 'configs/sharpa_w01_tacmap_compliant_preheld_v1.json',
                 'configs/sharpa_w01_hard_case.json', 'configs/sharpa_w01_natural_v6c_replay_cases.json',
                 'configs/sharpa_w01_tacmap_cameras.json']:
        paths.add(ROOT / name)
    for folder in ['src', 'analysis']:
        for name in (ROOT / folder).glob('*.py'):
            if folder == 'src' or name.name.startswith('validate_'):
                paths.add(name)
    for folder in ['scripts', 'review', 'observation_checks', 'visualizations', 'per_target_replay']:
        for p in (BASE / folder).rglob('*'):
            if '__pycache__' not in p.parts and p.suffix in {'.py', '.json', '.csv', '.png'}:
                paths.add(p)
    for p in (BASE / 'single_blade/run_2x2').glob('*'):
        if p.suffix in {'.json', '.png', '.csv'}:
            paths.add(p)
    for name in ['compliant_v1_full_validation_v2', 'compliant_v1_replay8']:
        for p in (BASE / 'runs' / name).glob('*.json'):
            if p.stat().st_size < 20_000_000:
                paths.add(p)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    entries = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)]
    manifest = {
        'scope': 'Review bundle, not a standalone simulator/assets distribution.',
        'root': str(ROOT), 'files': entries,
        'raw_data': 'Full NPZ, scene USD and native videos remain under sharpa_w01_tactile_validation/improvement_validation/runs; companion videos are delivered separately.',
        'status': 'Experimental compliant workflow: fixed case 1/1; matching-case replay 5/8 versus rigid 8/8. Raw official multi-target output contains confirmed false acceptance; per-target replay is a separate adaptation.'}
    manifest_path = BASE / '交付文件校验.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    output = BASE / 'W01_触觉改进_代码与检验.zip'
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in sorted(paths | {manifest_path}):
            archive.write(p, str(p.relative_to(ROOT)))
    print(json.dumps({'path': str(output), 'bytes': output.stat().st_size,
                      'files': len(paths) + 1,
                      'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
