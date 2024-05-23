import argparse
from pathlib import Path
import math
import shutil
import subprocess
import json


def run_bash_command(command: str) -> str:
    """
    Run a bash command and return the output as a string
    """
    print(f"Running command: >>>{command}<<<")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, _ = process.communicate()
    return output.decode("utf-8")


def run_jlaunch_commands(jlaunch_commands: Path) -> Path:
    """
    Run all the jlaunch commands in the given file and save their outputs to
    a file called "lauch_all.sh" in the same directory as the jlaunch commands.
    """

    assert jlaunch_commands.is_file(), f"Jlaunch commands file {jlaunch_commands} does not exist"

    jlaunch_results_path = jlaunch_commands.parent / "launch_all.sh"

    if jlaunch_results_path.exists():
        jlaunch_results_path.unlink()

    run_bash_command(f"bash {jlaunch_commands} > {jlaunch_results_path}")

    return jlaunch_results_path


def build_config(
    sequence_name : str, base_config: Path, config_file_save_path: Path, sequence_length : int | None
) -> None:
    assert base_config.is_file(), f"Config file {base_config} does not exist"
    custom_config_content = f"""_base_ = "{base_config}"
test_dataset = dict(args=dict(log_subset=["{sequence_name}"]), subsequence_length={sequence_length}))
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

    jlaunch_script = Path("data_prep_scripts/jlaunch.py").absolute()
    test_script = Path("test_pl.py").absolute()

    jlaunch_content = f"""#!/bin/bash
python {jlaunch_script} backend-{backend} "python {test_script} {job_config}" --name {job_name} {' '.join(jlaunch_args)}
"""

    with open(jlaunch_save_path, "w") as f:
        f.write(jlaunch_content)


def build_split(
    sequence_name: str,
    base_config: Path,
    launch_files_dir: Path,
    backend: str,
    job_name: str,
    sequence_length : int,
    jlaunch_args: list[str],
) -> Path:
    """
    Build the jlaunch file for a single job
    """

    job_dir = launch_files_dir / f"job_{sequence_name}"

    # Remove the job directory if it already exists
    if job_dir.exists():
        shutil.rmtree(job_dir)

    job_dir.mkdir(exist_ok=True, parents=True)

    job_config = job_dir / f"config.py"
    build_config(sequence_name, base_config, job_config, sequence_length)
    jlaunch_config = job_dir / "jlaunch.sh"
    build_jlaunch(job_config, jlaunch_config, backend, job_name, jlaunch_args)

    return jlaunch_config


def build_jlaunch_commands(jlaunches: list[Path], launch_files_dir: Path) -> Path:
    """
    Write a bash script to launch all jobs
    """

    launch_all_path = launch_files_dir / "jlaunch_commands.sh"

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
    sequence_file : Path,
    launch_files_dir: Path,
    backend: str,
    base_name: str,
    jlaunch_args: list[str],
):
    assert base_config.is_file(), f"Config file {base_config} does not exist"
    assert sequence_file.is_file(), f"Sequence file {sequence_file} does not exist"

    existing_folders = [e for e in launch_files_dir.iterdir() if e.is_dir()]
    # Make the launch files directory absolute, and under the base name.
    launch_files_dir = launch_files_dir.absolute() / f"{len(existing_folders):06d}_{base_name}"

    # Create a directory to store the jlaunch files
    launch_files_dir.mkdir(exist_ok=True, parents=True)

    # load the JSON from the sequence file
    with open(sequence_file, "r") as f:
        sequence_lengths = json.load(f)
    

    # Create a jlaunch file for each job
    jlaunch_files = [
        build_split(
            sequence_name,
            base_config,
            launch_files_dir,
            backend,
            f"{base_name}_{sequence_name}",
            sequence_length,
            jlaunch_args,
        )
        for sequence_name, sequence_length in sequence_lengths.items()
    ]

    jlaunch_commands = build_jlaunch_commands(jlaunch_files, launch_files_dir)
    print(f"jlaunch commands written to {jlaunch_commands}")
    print(f"Running jlaunch commands...")
    run_all_commands = run_jlaunch_commands(jlaunch_commands)
    print(f"Run all commands written to {run_all_commands} to launch jobs.")


if __name__ == "__main__":
    """
    This script takes a base config file, the number of jobs to split it into, and creates a directory
    to store each config split and a jlaunch file to run each split. It then writes a bash script to run
    all the jlaunch commands.

    jlaunch is then run on the bash script to launch all the jobs on the appropriate backend 
    (this may involve the user having to run another command to launch the jobs on the backend)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("base_config", type=Path)
    parser.add_argument("sequence_file", type=Path)
    parser.add_argument("launch_files_dir", type=Path)
    parser.add_argument("backend", type=str, choices=["slurm", "ngc"])
    parser.add_argument("base_name", type=str)
    parser.add_argument("jlaunch_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    base_config = args.base_config
    sequence_file = args.sequence_file
    launch_files_dir = args.launch_files_dir
    backend = args.backend
    base_name = args.base_name
    jlaunch_args = list(args.jlaunch_args)

    # Print arguments
    print("Parsed args:")
    print(f"\tbase_config: {base_config}")
    print(f"\tsequence_file: {sequence_file}")
    print(f"\tlaunch_files_dir: {launch_files_dir}")
    print(f"\tbackend: {backend}")
    print(f"\tbase_name: {base_name}")
    print(f"\tjlaunch_args: {jlaunch_args}")



    build_splits(
        base_config,
        sequence_file,
        launch_files_dir,
        backend,
        base_name,
        jlaunch_args,
    )
