import argparse
from pathlib import Path
import math
import shutil


def build_config(
    job_idx: int, num_jobs: int, base_config: Path, config_file_save_path: Path
) -> None:
    assert base_config.is_file(), f"Config file {base_config} does not exist"
    assert job_idx >= 0, f"Job index must be non-negative"
    assert num_jobs > 0, f"Number of jobs must be positive"
    assert job_idx < num_jobs, f"Job index must be less than number of jobs"

    custom_config_content = f"""_base_ = "{base_config}"
test_dataset = dict(args=dict(split=dict(split_idx={job_idx}, num_splits={num_jobs})))
"""
    with open(config_file_save_path, "w") as f:
        f.write(custom_config_content)


def build_jlaunch(
    job_config: Path,
    jlaunch_save_path: Path,
    backend: str,
    job_name: str,
    jlaunch_args: list[str],
):
    assert job_config.is_file(), f"Config file {job_config} does not exist"

    jlaunch_content = f"""#!/bin/bash
python data_prep_scripts/jlaunch.py backend-{backend} "python test_pl.py {job_config}" --name {job_name} {' '.join(jlaunch_args)}
"""

    with open(jlaunch_save_path, "w") as f:
        f.write(jlaunch_content)


def build_split(
    idx: int,
    num_jobs: int,
    base_config: Path,
    launch_files_dir: Path,
    backend: str,
    job_name: str,
    jlaunch_args: list[str],
) -> Path:
    """
    Build the jlaunch file for a single job
    """

    job_dir = launch_files_dir / f"job_{idx:06d}"

    # Remove the job directory if it already exists
    if job_dir.exists():
        shutil.rmtree(job_dir)

    job_dir.mkdir(exist_ok=True, parents=True)

    job_config = job_dir / f"config.py"
    build_config(idx, num_jobs, base_config, job_config)
    jlaunch_config = job_dir / "jlaunch.sh"
    build_jlaunch(job_config, jlaunch_config, backend, job_name, jlaunch_args)

    return jlaunch_config


def build_launch_all(jlaunches: list[Path], launch_files_dir: Path) -> Path:
    """
    Write a bash script to launch all jobs
    """

    launch_all_path = launch_files_dir / "launch_all.sh"

    launch_all_content = f"""#!/bin/bash
"""
    for jlaunch in jlaunches:
        assert jlaunch.is_file(), f"Jlaunch file {jlaunch} does not exist"
        launch_all_content += f"bash {jlaunch}\n"

    # Remove the launch all file if it already exists
    if launch_all_path.exists():
        launch_all_path.unlink()
    with open(launch_all_path, "w") as f:
        f.write(launch_all_content)

    return launch_all_path


def build_splits(
    base_config: Path,
    num_jobs: int,
    launch_files_dir: Path,
    backend: str,
    base_name: str,
    jlaunch_args: list[str],
):
    assert base_config.is_file(), f"Config file {base_config} does not exist"
    assert num_jobs > 0, f"Number of jobs must be positive"

    # Make the launch files directory absolute
    launch_files_dir = launch_files_dir.absolute()

    # Create a directory to store the jlaunch files
    launch_files_dir.mkdir(exist_ok=True, parents=True)

    # Create a jlaunch file for each job
    jlaunch_files = [
        build_split(
            idx,
            num_jobs,
            base_config,
            launch_files_dir,
            backend,
            f"{base_name}_{idx:06d}",
            jlaunch_args,
        )
        for idx in range(num_jobs)
    ]

    launch_all = build_launch_all(jlaunch_files, launch_files_dir)
    print(f"Launch all script written to {launch_all}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("base_config", type=Path)
    parser.add_argument("num_jobs", type=int)
    parser.add_argument("launch_files_dir", type=Path)
    parser.add_argument("backend", type=str, choices=["slurm", "ngc"])
    parser.add_argument("base_name", type=str)
    parser.add_argument("jlaunch_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    jlaunch_args = list(args.jlaunch_args)
    build_splits(
        args.base_config,
        args.num_jobs,
        args.launch_files_dir,
        args.backend,
        args.base_name,
        jlaunch_args,
    )
