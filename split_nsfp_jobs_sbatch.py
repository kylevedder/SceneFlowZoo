import argparse
from pathlib import Path
import math
import shutil

# Get path to argoverse lidar dataset and number of sequences per job
parser = argparse.ArgumentParser()
parser.add_argument('lidar_path', type=Path)
parser.add_argument('sequences_per_job', type=int)
parser.add_argument('base_config', type=Path)
parser.add_argument('--reset_config_dir', action='store_true')
parser.add_argument('--configs_path',
                    type=Path,
                    default=Path("./nsfp_split_configs"))
parser.add_argument('--runtime_mins', type=int, default=180)
parser.add_argument('--job_prefix', type=str, default='nsfp')
parser.add_argument('--elevated', action='store_true')
parser.add_argument('--job_subset_file', type=Path, default=None)
args = parser.parse_args()

assert args.lidar_path.is_dir(), f"Path {args.lidar_path} is not a directory"
assert args.sequences_per_job > 0, f"Number of sequences per job must be positive"
assert args.base_config.is_file(
), f"Config file {args.base_config} does not exist"

assert args.runtime_mins > 0, f"Runtime must be positive"

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

if args.job_subset_file is not None:
    assert args.job_subset_file.is_file(
    ), f"Job subset file {args.job_subset_file} does not exist"
    print(f"Using job subset file {args.job_subset_file} to filter jobs")

    def parse_line(line: str) -> str:
        if ' ' not in line:
            return line.strip()
        # Get the first column
        return line.split(' ')[0].strip()

    with open(args.job_subset_file, "r") as f:
        valid_jobs = [parse_line(l) for l in f.readlines()]
    valid_jobs = set(valid_jobs)

    # Filter out jobs that are not in the subset
    job_sequence_names_lst = [[job for job in sequence if job in valid_jobs]
                              for sequence in job_sequence_names_lst]

    # Filter out empty sequences
    job_sequence_names_lst = [
        sequence for sequence in job_sequence_names_lst if len(sequence) > 0
    ]

    print(f"Filtered to {len(job_sequence_names_lst)} jobs")

# sequence_names_set = set(f for seqs in job_sequence_names_lst for f in seqs)
# assert len(sequence_names_set) == len(
#     sequence_folders), "Some sequences are missing from jobs"

configs_path = args.configs_path
if args.reset_config_dir:
    if configs_path.exists():
        shutil.rmtree(configs_path)
    configs_path.mkdir(exist_ok=False)
else:
    configs_path.mkdir(exist_ok=True, parents=True)


def get_runtime_format(runtime_mins):
    hours = runtime_mins // 60
    minutes = runtime_mins % 60
    return f"{hours:02d}:{minutes:02d}:00"


def make_config(i, job_sequence_names):
    config_path = configs_path / f"nsfp_split_{i:06d}.py"
    config_file_content = f"""
_base_ = '../{args.base_config}'
test_loader = dict(args=dict(log_subset={job_sequence_names}))
"""
    with open(config_path, "w") as f:
        f.write(config_file_content)


def make_sbatch():
    current_working_dir = Path.cwd().absolute()
    sbatch_path = configs_path / f"sbatch.bash"
    if args.elevated:
        qos = "ee-med"
        partition = "eaton-compute"
    else:
        qos = "batch"
        partition = "batch"
    sbatch_file_content = f"""#!/bin/bash
#SBATCH --job-name={args.job_prefix}
#SBATCH --output={configs_path}/nsfp_%a.out
#SBATCH --error={configs_path}/nsfp_%a.err
#SBATCH --time={get_runtime_format(args.runtime_mins)}"""
    if args.elevated:
        sbatch_file_content += f"""
#SBATCH --qos={qos}
#SBATCH --partition={partition}"""
    sbatch_file_content += f"""
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=12G
#SBATCH --cpus-per-gpu=2
#SBATCH --exclude=kd-2080ti-2.grasp.maas
#SBATCH --array=0-{len(job_sequence_names_lst) - 1}
#SBATCH --container-mounts=../../datasets/:/efs/,{current_working_dir}:/project
#SBATCH --container-image={current_working_dir / "kylevedder_offline_sceneflow_latest.sqsh"}

export MY_RUN_ID=$(printf "%06d" $SLURM_ARRAY_TASK_ID)
python test_pl.py {configs_path}/nsfp_split_$MY_RUN_ID.py; echo 'done' > {configs_path}/nsfp_$MY_RUN_ID.done
"""
    with open(sbatch_path, "w") as f:
        f.write(sbatch_file_content)


for i, job_sequence_names in enumerate(job_sequence_names_lst):
    make_config(i, job_sequence_names)
    if i % 100 == 0:
        print(f"Made configs for {i} jobs")
print(f"Made total of {len(job_sequence_names_lst)} jobs")

make_sbatch()

print(f"Config files written to {configs_path.absolute()}")
