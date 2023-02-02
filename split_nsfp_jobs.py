import argparse
from pathlib import Path
import math
import shutil

# Get path to argoverse lidar dataset and number of sequences per job
parser = argparse.ArgumentParser()
parser.add_argument('lidar_path', type=Path)
parser.add_argument('sequences_per_job', type=int)
parser.add_argument('base_config', type=Path)
args = parser.parse_args()

assert args.lidar_path.is_dir(), f"Path {args.lidar_path} is not a directory"
assert args.sequences_per_job > 0, f"Number of sequences per job must be positive"
assert args.base_config.is_file(), f"Config file {args.base_config} does not exist"

sequence_folders = sorted([c for c in args.lidar_path.glob("*") if c.is_dir()],
                          key=lambda x: x.name.lower())
num_jobs = math.ceil(len(sequence_folders) / args.sequences_per_job)

print(f"Splitting {len(sequence_folders)} sequences into {num_jobs} jobs")


job_sequence_names_lst = []
for i in range(num_jobs):
    start = i * args.sequences_per_job
    end = min(start + args.sequences_per_job, len(sequence_folders))
    job_sequence_folders = sequence_folders[start:end]
    job_sequence_names = [f.name for f in job_sequence_folders]
    job_sequence_names_lst.append(job_sequence_names)

sequence_names_set = set(f for seqs in job_sequence_names_lst for f in seqs)
assert len(sequence_names_set) == len(sequence_folders), "Some sequences are missing from jobs"


configs_path = Path("./nsfp_split_configs")
if configs_path.exists():
    shutil.rmtree(configs_path)
configs_path.mkdir(exist_ok=False)
for i, job_sequence_names in enumerate(job_sequence_names_lst):
    config_path = configs_path / f"nsfp_split_{i}.py"
    config_file_content = f"""
_base_ = '{args.base_config}'
test_loader = dict(args=dict(log_subset={job_sequence_names}))
"""
    with open(config_path, "w") as f:
        f.write(config_file_content)

print(f"Config files written to {configs_path.absolute()}")

# Create SBATCH script
sbatch_path = configs_path / "run_nsfp_split.sh"
sbatch_file_content = f"""#!/bin/bash
#SBATCH --job-name=nsfp_split
#SBATCH --output={configs_path}/nsfp_split_%j.out
#SBATCH --error={configs_path}/nsfp_split_%j.err
#SBATCH --time=3:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --array=0-{num_jobs-1}

docker run --gpus=all --rm -v `pwd`:/project -v datasets/:/efs kylevedder/offline_sceneflow:latest bash -c "echo configs/nsfp_split_configs/nsfp_split_\$SLURM_ARRAY_TASK_ID.py"
"""
with open(sbatch_path, "w") as f:
    f.write(sbatch_file_content)