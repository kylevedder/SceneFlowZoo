from pathlib import Path
import json
import pickle
import subprocess
import numpy as np
import pandas as pd

from typing import Union


def _compute_size_metric(filepath: Path):
    size = filepath.stat().st_size
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.2f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.2f} MB"
    else:
        return f"{size / (1024 * 1024 * 1024):.2f} GB"


def load_txt(filepath: Path, verbose: bool = True):
    filepath = Path(filepath)
    assert filepath.exists(), f"{filepath} does not exist"
    if verbose:
        print(f"Loading {filepath} of size {_compute_size_metric(filepath)}")
    with open(filepath, "r") as f:
        return f.read()


def save_txt(filepath: Path, txt, verbose: bool = True):
    filepath = Path(filepath)
    if verbose:
        print(f"Saving {filepath}", end="")
    if filepath.exists():
        filepath.unlink()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        f.write(txt)
    if verbose:
        print(f"\rSaved {filepath} of size {_compute_size_metric(filepath)}")


def load_npz(filepath: Path, verbose: bool = True):
    filepath = Path(filepath)
    assert filepath.exists(), f"{filepath} does not exist"
    if verbose:
        print(f"Loading {filepath} of size {_compute_size_metric(filepath)}")
    return np.load(filepath, allow_pickle=True)


def save_npz(filepath: Path, npz, verbose: bool = True):
    filepath = Path(filepath)
    if verbose:
        print(f"Saving {filepath}", end="")
    if filepath.exists():
        filepath.unlink()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(filepath, **npz)
    if verbose:
        print(f"\rSaved {filepath} of size {_compute_size_metric(filepath)}")


def load_npy(filepath: Path, verbose: bool = True):
    filepath = Path(filepath)
    assert filepath.exists(), f"{filepath} does not exist"
    if verbose:
        print(f"Loading {filepath} of size {_compute_size_metric(filepath)}")
    return np.load(filepath, allow_pickle=True)


def save_npy(filepath: Path, npy, verbose: bool = True):
    filepath = Path(filepath)
    if verbose:
        print(f"Saving {filepath}", end="")
    if filepath.exists():
        filepath.unlink()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath, npy)
    if verbose:
        print(f"\rSaved {filepath} of size {_compute_size_metric(filepath)}")


def load_pickle(filepath: Path, verbose: bool = True):
    filepath = Path(filepath)
    assert filepath.exists(), f"{filepath} does not exist"
    if verbose:
        print(f"Loading {filepath} of size {_compute_size_metric(filepath)}")
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_pickle(filepath: Path, pkl, verbose: bool = True):
    filepath = Path(filepath)
    if verbose:
        print(f"Saving {filepath}", end="")
    if filepath.exists():
        filepath.unlink()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(pkl, f)
    if verbose:
        print(f"\rSaved {filepath} of size {_compute_size_metric(filepath)}")


def load_json(filename: Path, verbose: bool = True):
    filename = Path(filename)
    assert filename.exists(), f"{filename} does not exist"
    if verbose:
        print(f"Loading {filename} of size {_compute_size_metric(filename)}")
    with open(filename) as f:
        return json.load(f)


def save_json(filename: Path, contents, indent=None, verbose: bool = True):
    if verbose:
        print(f"Saving {filename}", end="")
    filename = Path(filename)
    if filename.exists():
        filename.unlink()
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        json.dump(contents, f, indent=indent)
    if verbose:
        print(f"\rSaved {filename} of size {_compute_size_metric(filename)}")


def load_csv(filename: Path, dtype=str, verbose: bool = True):
    print(f"Loading {filename} of size {_compute_size_metric(filename)}")
    with open(filename) as f:
        return [[dtype(e.strip()) for e in line.strip().split(",")] for line in f.readlines()]


def save_csv(filename: Path, contents: list, verbose: bool = True):
    filename = Path(filename)
    if verbose:
        print(f"Saving {filename}", end="")
    if filename.exists():
        filename.unlink()
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        for line in contents:
            f.write(f'{",".join([str(e) for e in line])}\n')
    if verbose:
        print(f"\rSaved {filename} of size {_compute_size_metric(filename)}")


def load_feather(filename: Path, verbose: bool = True):
    filename = Path(filename)
    assert filename.exists(), f"{filename} does not exist"
    if verbose:
        print(f"Loading {filename} of size {_compute_size_metric(filename)}")
    return pd.read_feather(filename)


def save_feather(filename: Path, contents : Union[pd.DataFrame, dict], verbose: bool = True):
    if isinstance(contents, dict):
        contents = pd.DataFrame(contents)
    filename = Path(filename)
    if verbose:
        print(f"Saving {filename}", end="")
    if filename.exists():
        filename.unlink()
    filename.parent.mkdir(parents=True, exist_ok=True)
    contents.to_feather(filename)
    if verbose:
        print(f"\rSaved {filename} of size {_compute_size_metric(filename)}")


def save_by_extension(filename: Path, contents, verbose: bool = True):
    filename = Path(filename)
    # Make parents if they don't exist
    filename.parent.mkdir(parents=True, exist_ok=True)
    if filename.suffix == ".txt":
        save_txt(filename, contents, verbose=verbose)
    elif filename.suffix == ".npz":
        save_npz(filename, contents, verbose=verbose)
    elif filename.suffix == ".npy":
        save_npy(filename, contents, verbose=verbose)
    elif filename.suffix == ".pkl":
        save_pickle(filename, contents, verbose=verbose)
    elif filename.suffix == ".json":
        save_json(filename, contents, verbose=verbose)
    elif filename.suffix == ".csv":
        save_csv(filename, contents, verbose=verbose)
    elif filename.suffix == ".feather":
        save_feather(filename, contents, verbose=verbose)
    else:
        raise ValueError(f"Unknown file extension: {filename.suffix}")


def load_by_extension(filename: Path, verbose: bool = True):
    filename = Path(filename)
    if filename.suffix == ".txt":
        return load_txt(filename, verbose=verbose)
    elif filename.suffix == ".npz":
        return load_npz(filename, verbose=verbose)
    elif filename.suffix == ".npy":
        return load_npy(filename, verbose=verbose)
    elif filename.suffix == ".pkl":
        return load_pickle(filename, verbose=verbose)
    elif filename.suffix == ".json":
        return load_json(filename, verbose=verbose)
    elif filename.suffix == ".csv":
        return load_csv(filename, verbose=verbose)
    elif filename.suffix == ".feather":
        return load_feather(filename, verbose=verbose)
    else:
        raise ValueError(f"Unknown file extension: {filename.suffix}")


def symlink_files(old_dir: Path, new_dir: Path, file_name_list: list):
    old_dir = Path(old_dir)
    assert old_dir.exists(), f"{old_dir} does not exist"
    new_dir = Path(new_dir)
    assert new_dir.exists(), f"{new_dir} does not exist"
    for fn in file_name_list:
        if (new_dir / fn).exists():
            (new_dir / fn).unlink()
        (new_dir / fn).symlink_to(old_dir / fn)


def run_cmd(cmd: str, return_stdout: bool = False):
    res = subprocess.run(cmd, shell=True, encoding="utf-8", capture_output=True)
    if res.returncode != 0:
        print(f"Error running command: {cmd}")
        print(res.stdout)
        print(res.stderr)
    else:
        lines = res.stdout.strip().splitlines()
        if len(lines) > 0:
            print(lines[0].strip())
            print("...")
            print(lines[-1].strip())
    assert res.returncode == 0, f"Command {cmd} failed with return code {res.returncode}"
    if return_stdout:
        return res.stdout
