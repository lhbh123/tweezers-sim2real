"""Execute an explicitly reviewed, workspace-local cleanup manifest.

No path discovery is performed on apply. Every file must match its recorded
size/mtime, remain inside this workspace, and not be used by a live process.
The manifest is an audit log, not a backup or an automatic retention policy.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def live_paths():
    result = set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cwd = (proc / 'cwd').resolve()
            if not cwd.is_relative_to(ROOT):
                continue
            args = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            for arg in args:
                if not arg or arg.startswith('-') or '\n' in arg:
                    continue
                p = Path(arg) if arg.startswith('/') else cwd / arg
                if p.exists() and p.resolve().is_relative_to(ROOT):
                    result.add(p.resolve())
            for fd in (proc / 'fd').iterdir():
                try:
                    p = fd.resolve()
                    if p.exists() and p.is_relative_to(ROOT):
                        result.add(p)
                except (OSError, RuntimeError):
                    pass
        except (OSError, RuntimeError):
            pass
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    data = json.loads(a.manifest.read_text())
    assert data['root'] == str(ROOT)
    assert data['status'] == 'reviewed_plan', 'Manifest already applied or not reviewed'
    active = live_paths()
    # Validate the entire plan before deleting the first file.
    for item in data['delete_files']:
        relative = Path(item['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        path = ROOT / relative
        assert path.resolve().is_relative_to(ROOT) and not path.is_symlink()
        assert path.is_file(), str(relative)
        st = path.stat()
        assert st.st_size == item['bytes'] and st.st_mtime_ns == item['mtime_ns'], str(relative)
        assert not any(path == q or (q.is_dir() and q != ROOT and path.is_relative_to(q)) for q in active), str(relative)
    for item in data['protected_hashes']:
        path = ROOT / item['path']
        assert hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() == item['sha256']
    print(json.dumps({'files': len(data['delete_files']), 'bytes': sum(x['bytes'] for x in data['delete_files']), 'apply': a.apply}), flush=True)
    if not a.apply:
        return
    for item in data['delete_files']:
        (ROOT / item['path']).unlink()
    removed_dirs = []
    for name in sorted(data['candidate_empty_directories'], key=lambda x: (-len(Path(x).parts), x)):
        path = ROOT / name
        assert path.resolve().is_relative_to(ROOT) and not path.is_symlink()
        try:
            path.rmdir()
            removed_dirs.append(name)
        except OSError:
            pass
    for item in data['protected_hashes']:
        assert hashlib.file_digest((ROOT / item['path']).open('rb'), 'sha256').hexdigest() == item['sha256']
    data['status'] = 'completed'
    data['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    data['deleted_files'] = len(data['delete_files'])
    data['deleted_bytes'] = sum(x['bytes'] for x in data['delete_files'])
    data['removed_directories'] = removed_dirs
    data['protected_hashes_verified_after_cleanup'] = True
    a.manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print('Cleanup completed; retained artifacts verified unchanged.', flush=True)


if __name__ == '__main__':
    main()
