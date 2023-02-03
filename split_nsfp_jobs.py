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



def make_config(i, job_sequence_names):
    config_path = configs_path / f"nsfp_split_{i:06d}.py"
    config_file_content = f"""
_base_ = '../{args.base_config}'
test_loader = dict(args=dict(log_subset={job_sequence_names}))
"""
    with open(config_path, "w") as f:
        f.write(config_file_content)
    

def make_srun(i):
    srun_path = configs_path / f"srun_{i:06d}.sh"
    srun_file_content = f"""#!/bin/bash
srun --gpus=1 --mem-per-gpu=12G --cpus-per-gpu=2 --time=03:00:00 --container_mounts=../../datasets/:/efs/,`pwd`:project --container-image=kylevedder/offline_sceneflow:latest bash -c "python test_pl.py {configs_path}/nsfp_split_{i:06d}.py; echo 'done' > {configs_path}/nsfp_{i:06d}.done"
"""
    with open(srun_path, "w") as f:
        f.write(srun_file_content)

def make_screen(i):
    screen_path = configs_path / f"screen_{i:06d}.sh"
    screen_file_content = f"""#!/bin/bash
echo "" > {configs_path}/nsfp_{i:06d}.out;
screen -L -Logfile {configs_path}/nsfp_{i:06d}.out -dmS nsfp_{i:06d} bash {configs_path}/srun_{i:06d}.sh
"""
    with open(screen_path, "w") as f:
        f.write(screen_file_content)

def make_runall():
    runall_path = configs_path / f"runall.sh"
    runall_file_content = f"""#!/bin/bash
for i in {configs_path / "screen_*.sh"}; do
    echo $i
    bash $i
done
"""
    with open(runall_path, "w") as f:
        f.write(runall_file_content)

for i, job_sequence_names in enumerate(job_sequence_names_lst):
    make_config(i, job_sequence_names)
    make_srun(i)
    make_screen(i)

make_runall()

print(f"Config files written to {configs_path.absolute()}")
