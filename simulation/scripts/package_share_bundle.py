"""Build a portable task source/asset bundle; omit caches and historical media."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache'}
SKIP_SUFFIXES = {'.pyc', '.pyo', '.log', '.partial', '.blend1'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_tree(source, target, suffixes=None):
    for path in sorted(source.rglob('*')):
        relative = path.relative_to(source)
        if not path.is_file() or any(p in SKIP_PARTS for p in relative.parts):
            continue
        if path.suffix in SKIP_SUFFIXES or (suffixes and path.suffix not in suffixes):
            continue
        if path.is_symlink():
            raise RuntimeError(f'Unexpected symlink: {path}')
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--external-assets', type=Path, default=ROOT.parent/'assets/ur5e_xhand_right')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='tweezers-share-') as temporary:
        bundle = Path(temporary)/'tweezers_xhand_w01'
        project = bundle/'tweezers'
        project.mkdir(parents=True)
        for directory in ['src', 'scripts', 'configs']:
            copy_tree(ROOT/directory, project/directory)
        copy_tree(ROOT/'analysis', project/'analysis', {'.py', '.sh', '.json', '.txt', '.md', '.csv'})
        for directory in ['sharpa_wave/official', 'sharpa_tactile',
                          'sharpa_tacmap_w01_matched_v2', 'ur5e_sharpa_w01_right', 'fonts']:
            copy_tree(ROOT/'assets'/directory, project/'assets'/directory)
        copy_tree(args.external_assets, bundle/'assets/ur5e_xhand_right')
        extra = [
            'TUTORIAL.md', 'README.md', 'README_XHAND.md', 'README_SHARPA_W01.md',
            'README_SHARPA_W01_NATURAL.md',
            'runs/middle_backup13_level_search/m9_c0_snapshot.json',
            'runs/pick_place15_ppo101_retreat80/iteration_00025.pt',
            'reports/environment_snapshot_20261019.json',
            'reports/pick_place_revision_final.json',
            'reports/pick_place15_frozen_candidate.json',
            'reports/pick_place15_training_evidence.json',
            'reports/sharpa_w01_natural_v6d_final.json',
            'reports/sharpa_w01_official_tacmap_final.json',
        ]
        for name in extra:
            target = project/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, target)
        # Keep the optional compliant task's diagnosis code and written report,
        # without shipping its large historical sensor arrays and videos.
        for directory in ['sharpa_w01_tactile_validation/improvement_validation/scripts',
                          'sharpa_w01_tactile_validation/improvement_validation/observation_checks']:
            copy_tree(ROOT/directory, project/directory, {'.py', '.sh', '.md', '.txt'})
        report = ROOT/'sharpa_w01_tactile_validation/improvement_validation/IMPROVEMENT_VALIDATION_REPORT.txt'
        if report.exists():
            shutil.copy2(report, project/report.relative_to(ROOT))

        (bundle/'READ_ME_FIRST.txt').write_text('''灵巧手使用镊子夹取方块：XHand + Sharpa Wave W01 分享包

解压：tar -xJf tweezers_xhand_w01_20261021.tar.xz
进入：cd tweezers_xhand_w01/tweezers
阅读：TUTORIAL.md

包内目录请保持原样：
  assets/ur5e_xhand_right/  外部组合机器人、UR5e/XHand 源网格和 USD 子层
  tweezers/               代码、教程、配置、W01 资产、策略及初始快照

两套机器人共用的外部资产已包含，无需再向原服务器索取。
首先按教程安装 Isaac Sim 5.0.0 / Isaac Lab v2.2.1，设置 ISAAC_PYTHON，
再执行教程第 4 节生成本机路径配置。原始 JSON 和历史记录保留开发路径，
运行时务必使用 xhand_local.json / sharpa_w01_local.json。
不要只把“tweezers”子目录单独发给别人，否则会丢失外部资产。

包含：两套任务主链路、训练/验证/视频工具源码、教程和说明、手模型及网格、
TacMap 原始/匹配表面与兼容源码、XHand 已训练权重、两套预持镊快照、
字体、原资产附带的许可证和来源记录，以及部分历史技术报告。
可选柔顺 W01 配置、快照和诊断源码也保留；教程默认使用刚性基线。

不包含：Isaac Sim 安装包、Python 虚拟环境、GPU 驱动、历史录像、完整实验
轨迹/触觉数组、人类参考视频、对话历史、Git 历史或代理/账号配置。
文档和历史开发脚本中指向未随包交付的旧实验数据/媒体的链接不可直接使用。
主教程的新运行命令会在本机生成新结果；旧数据审计或旧媒体重制需另取对应数据。
这不是全项目历史备份。基础任务从已经持镊开始，不包含从桌上捡起镊子。

完整性校验（在 tweezers_xhand_w01 目录执行）：
  python3 verify_package.py
MANIFEST.json 记录逐文件大小及 SHA-256。先校验再生成本机配置。
PACKAGE_VALIDATION.json 记录打包时的静态检查，不代表在新机器重新跑过仿真。
源码、资产沿用各自原始许可证；本包不为第三方资产赋予新的许可。
''', encoding='utf-8')
        (bundle/'verify_package.py').write_text('''"""Verify original package files; locally generated extra files are allowed."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "MANIFEST.json").read_text())
bad = []
for name, metadata in manifest["files"].items():
    path = root / name
    if not path.is_file() or path.stat().st_size != metadata["bytes"]:
        bad.append(name)
    elif hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        bad.append(name)
if bad:
    raise SystemExit("文件缺失或被修改：\\n" + "\\n".join(bad))
print("完整性检查通过：", len(manifest["files"]), "个原始文件")
''', encoding='utf-8')

        # Check references within a genuinely relocated, copied asset layout.
        mesh_counts = {}
        for asset in [bundle/'assets/ur5e_xhand_right', project/'assets/ur5e_sharpa_w01_right']:
            tree = ET.parse(asset/'robot.urdf')
            references = {node.get('filename') for node in tree.findall('.//mesh')}
            for reference in references:
                path = (asset/reference).resolve()
                if not path.is_relative_to(bundle) or not path.is_file():
                    raise RuntimeError(f'Mesh escapes package or is missing: {reference}')
            mesh_counts[asset.relative_to(bundle).as_posix()] = len(references)
        doc = (project/'TUTORIAL.md').read_text()
        python_snippets = []
        for language, body in re.findall(r'```([^\n]*)\n(.*?)```', doc, re.S):
            if language == 'bash':
                subprocess.run(['bash', '-n'], input=body, text=True, check=True)
                python_snippets += re.findall(r"<<'PY'\n(.*?)\nPY", body, re.S)
        for snippet in python_snippets:
            ast.parse(snippet)
        generated_names = ['xhand_local.json', 'sharpa_w01_local.json']
        if any((project/'configs'/n).exists() for n in generated_names):
            raise RuntimeError('Do not package machine-specific local pose files')
        # Exercise the documented config generation/preflight in another copy.
        with tempfile.TemporaryDirectory(prefix='tweezers-package-check-') as check_dir:
            check_root = Path(check_dir)
            check_project = check_root/'tweezers'
            shutil.copytree(project, check_project)
            shutil.copytree(bundle/'assets', check_root/'assets')
            for marker in ['xhand_asset =', 'required = [']:
                snippet = next(s for s in python_snippets if marker in s)
                result = subprocess.run([sys.executable, '-c', snippet], cwd=check_project,
                                        capture_output=True, text=True, check=True)
                print(result.stdout.strip(), flush=True)
            configs = [json.loads((check_project/'configs'/n).read_text()) for n in generated_names]
            assert len(configs[0]['mechanical_self_collision_pairs']) == 2
            assert len(configs[1]['mechanical_self_collision_pairs']) == 7
        validation = {'mesh_references_resolve_within_bundle': mesh_counts,
                      'tutorial_shell_and_python_syntax': 'passed',
                      'tutorial_config_generation_and_preflight_in_relocated_copy': 'passed',
                      'hand_mechanism_filter_counts': {'xhand': 2, 'w01': 7},
                      'simulation_rerun': False, 'clean_machine_installation_tested': False}
        (bundle/'PACKAGE_VALIDATION.json').write_text(json.dumps(validation, indent=2)+'\n')
        manifest = {p.relative_to(bundle).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                    for p in sorted(bundle.rglob('*')) if p.is_file()}
        (bundle/'MANIFEST.json').write_text(json.dumps({'schema_version': 1, 'files': manifest},
                                                     ensure_ascii=False, indent=2)+'\n')
        print(f'Packing {len(manifest)} files, {sum(x["bytes"] for x in manifest.values())} bytes', flush=True)
        with tarfile.open(output, 'w:xz', preset=6) as archive:
            archive.add(bundle, arcname=bundle.name, recursive=True)
        # Verify the actual archive's entries and bytes, not only staging files.
        checked = set()
        with tarfile.open(output, 'r:xz') as archive:
            for member in archive:
                if member.isdir():
                    continue
                if not member.isfile():
                    raise RuntimeError(f'Unexpected non-regular member: {member.name}')
                relative = Path(member.name).relative_to(bundle.name).as_posix()
                if '..' in Path(relative).parts:
                    raise RuntimeError('Unsafe archive member')
                content = archive.extractfile(member).read()
                if relative == 'MANIFEST.json':
                    assert json.loads(content)['files'] == manifest
                else:
                    assert relative in manifest
                    assert len(content) == manifest[relative]['bytes']
                    assert hashlib.sha256(content).hexdigest() == manifest[relative]['sha256']
                    checked.add(relative)
        assert checked == set(manifest)
        result = {'file': str(output), 'bytes': output.stat().st_size, 'sha256': sha(output),
                  'manifest_files': len(manifest), 'archive_hash_validation': 'passed',
                  'validation': validation}
        output.with_suffix(output.suffix+'.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        output.with_suffix(output.suffix+'.sha256').write_text(result['sha256']+'  '+output.name+'\n')
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
