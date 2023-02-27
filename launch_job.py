import argparse
from pathlib import Path
import time
from loader_utils import run_cmd

parser = argparse.ArgumentParser()
parser.add_argument('command', type=str)
parser.add_argument('--job_dir', type=Path, default="./job_dir/")
parser.add_argument('--num_gpus', type=int, default=1)
parser.add_argument('--cpus_per_gpu', type=int, default=2)
parser.add_argument('--mem_per_gpu', type=int, default=12)
parser.add_argument('--runtime_mins', type=int, default=180)
parser.add_argument('--runtime_hours', type=int, default=None)
parser.add_argument('--job_name', type=str, default='ff3d')
parser.add_argument('--qos', type=str, default='ee-med')
parser.add_argument('--partition', type=str, default='eaton-compute')
parser.add_argument('--dry_run', action='store_true')
parser.add_argument('--use_srun', action='store_true')
parser.add_argument('--blacklist_substring', type=str, default=None)
args = parser.parse_args()

num_prior_jobs = len(list(args.job_dir.glob("*")))
jobdir_path = args.job_dir / f"{num_prior_jobs:06d}"
jobdir_path.mkdir(exist_ok=True, parents=True)
job_runtime_mins = args.runtime_mins if args.runtime_hours is None else args.runtime_hours * 60


def load_available_nodes():
    res = run_cmd("sinfo --Node | awk '{print $1}' | tail +2",
                  return_stdout=True)
    available_nodes = res.split('\n')
    return [e.strip() for e in available_nodes]


node_blacklist = []
if args.blacklist_substring is not None:
    node_blacklist = [
        node for node in load_available_nodes()
        if args.blacklist_substring in node
    ]


def get_runtime_format(runtime_mins):
    hours = runtime_mins // 60
    minutes = runtime_mins % 60
    return f"{hours:02d}:{minutes:02d}:00"


def make_command_file(command):
    command_path = jobdir_path / f"command.sh"
    command_file_content = f"""#!/bin/bash
{command}
"""
    with open(command_path, "w") as f:
        f.write(command_file_content)


def make_srun():
    srun_path = jobdir_path / f"srun.sh"
    docker_image_path = Path(
        "kylevedder_offline_sceneflow_latest.sqsh").absolute()
    assert docker_image_path.is_file(
    ), f"Docker image {docker_image_path} squash file does not exist"
    srun_file_content = f"""#!/bin/bash
srun --gpus={args.num_gpus} --nodes=1 --mem-per-gpu={args.mem_per_gpu}G --cpus-per-gpu={args.cpus_per_gpu} --time={get_runtime_format(job_runtime_mins)} --exclude=kd-2080ti-2.grasp.maas --job-name={args.job_name} --qos={args.qos} --partition={args.partition} --container-mounts=../../datasets/:/efs/,`pwd`:/project --container-image={docker_image_path} bash {jobdir_path}/command.sh
"""
    with open(srun_path, "w") as f:
        f.write(srun_file_content)


def make_sbatch():
    current_working_dir = Path.cwd().absolute()
    sbatch_path = jobdir_path / f"sbatch.bash"
    docker_image_path = Path(
        "kylevedder_offline_sceneflow_latest.sqsh").absolute()
    assert docker_image_path.is_file(
    ), f"Docker image {docker_image_path} squash file does not exist"
    sbatch_file_content = f"""#!/bin/bash
#SBATCH --job-name={args.job_name}
#SBATCH --qos={args.qos}
#SBATCH --partition={args.partition}
#SBATCH --nodes=1
#SBATCH --output={jobdir_path}/job.out
#SBATCH --error={jobdir_path}/job.err
#SBATCH --time={get_runtime_format(job_runtime_mins)}
#SBATCH --gpus={args.num_gpus}
#SBATCH --mem-per-gpu={args.mem_per_gpu}G
#SBATCH --cpus-per-gpu={args.cpus_per_gpu}
#SBATCH --exclude={','.join(node_blacklist)}
#SBATCH --container-mounts=../../datasets/:/efs/,{current_working_dir}:/project
#SBATCH --container-image={docker_image_path}

bash {jobdir_path}/command.sh; echo 'done' > {jobdir_path}/job.done
"""
    with open(sbatch_path, "w") as f:
        f.write(sbatch_file_content)


def make_screen():
    screen_path = jobdir_path / f"screen.sh"
    screen_file_content = f"""#!/bin/bash
screen -L -Logfile {jobdir_path}/stdout.log -dmS {args.job_name} bash {jobdir_path}/srun.sh
"""
    with open(screen_path, "w") as f:
        f.write(screen_file_content)


make_command_file(args.command)
if args.use_srun:
    make_srun()
    make_screen()
else:
    make_sbatch()
if not args.dry_run:
    if args.use_srun:
        run_cmd(f"bash {jobdir_path}/screen.sh")
    else:
        run_cmd(f"sbatch {jobdir_path}/sbatch.bash")

print(f"Config files written to {jobdir_path.absolute()}")
