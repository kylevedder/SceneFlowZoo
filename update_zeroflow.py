from pathlib import Path
import argparse
import shutil

# Get optional path to the ZeroFlow public repo.
parser = argparse.ArgumentParser()
parser.add_argument("zeroflow_repo", type=Path)
args = parser.parse_args()

assert args.zeroflow_repo.exists(), f"{args.zeroflow_repo} does not exist"

current_path = Path().resolve()

# Use the entries in the ZeroFlow repo to grab the files / folders we need from this repo.
for zeroflow_path in args.zeroflow_repo.glob("*"):

    # Copy from this repo to the ZeroFlow repo.
    if zeroflow_path.name == ".git":
        print(f"Skipping {zeroflow_path}")
        continue
    if zeroflow_path.is_dir():
        print(f"Copying directory {zeroflow_path.name} to {zeroflow_path}")
        shutil.rmtree(zeroflow_path, ignore_errors=True)
        shutil.copytree(current_path / zeroflow_path.name, zeroflow_path)
    else:
        print(f"Copying file {zeroflow_path.name} to {zeroflow_path}")
        shutil.copy(current_path / zeroflow_path.name, zeroflow_path)
