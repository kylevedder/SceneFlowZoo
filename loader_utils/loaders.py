from pathlib import Path
import json
import pickle
import subprocess


def _compute_size_metric(filepath: Path):
    size = filepath.stat().st_size
    if size < 1024:
        return f'{size} B'
    elif size < 1024 * 1024:
        return f'{size / 1024:.2f} KB'
    elif size < 1024 * 1024 * 1024:
        return f'{size / (1024 * 1024):.2f} MB'
    else:
        return f'{size / (1024 * 1024 * 1024):.2f} GB'


def load_pickle(filepath: Path):
    filepath = Path(filepath)
    assert filepath.exists(), f'{filepath} does not exist'
    print(f'Loading {filepath} of size {_compute_size_metric(filepath)}')
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(filepath: Path, pkl):
    filepath = Path(filepath)
    print(f'Saving {filepath}', end='')
    if filepath.exists():
        filepath.unlink()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(pkl, f)
    print(f"\rSaved {filepath} of size {_compute_size_metric(filepath)}")


def load_json(filename: Path, verbose: bool = True):
    filename = Path(filename)
    assert filename.exists(), f'{filename} does not exist'
    if verbose:
        print(f'Loading {filename} of size {_compute_size_metric(filename)}')
    with open(filename) as f:
        return json.load(f)


def save_json(filename: Path, contents, indent=None):
    print(f'Saving {filename}', end='')
    filename = Path(filename)
    if filename.exists():
        filename.unlink()
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(contents, f, indent=indent)

    print(f"\rSaved {filename} of size {_compute_size_metric(filename)}")


def load_csv(filename: Path, dtype=str):
    print(f'Loading {filename} of size {_compute_size_metric(filename)}')
    with open(filename) as f:
        return [[dtype(e.strip()) for e in line.strip().split(',')]
                for line in f.readlines()]


def save_csv(filename: Path, contents: list):
    print(f'Saving {filename}', end='')
    if filename.exists():
        filename.unlink()
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as f:
        for line in contents:
            f.write(f'{",".join([str(e) for e in line])}\n')
    print(f"\rSaved {filename} of size {_compute_size_metric(filename)}")


def symlink_files(old_dir: Path, new_dir: Path, file_name_list: list):
    old_dir = Path(old_dir)
    assert old_dir.exists(), f'{old_dir} does not exist'
    new_dir = Path(new_dir)
    assert new_dir.exists(), f'{new_dir} does not exist'
    for fn in file_name_list:
        if (new_dir / fn).exists():
            (new_dir / fn).unlink()
        (new_dir / fn).symlink_to(old_dir / fn)


def run_cmd(cmd: str, return_stdout: bool = False):
    res = subprocess.run(cmd,
                         shell=True,
                         encoding='utf-8',
                         capture_output=True)
    if res.returncode != 0:
        print(f'Error running command: {cmd}')
        print(res.stdout)
        print(res.stderr)
    else:
        lines = res.stdout.strip().splitlines()
        if len(lines) > 0:
            print(lines[0].strip())
            print('...')
            print(lines[-1].strip())
    assert res.returncode == 0, f'Command {cmd} failed with return code {res.returncode}'
    if return_stdout:
        return res.stdout