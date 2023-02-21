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
parser.add_argument('--dry_run', action='store_true')
args = parser.parse_args()

jobdir_path = args.job_dir / f"{time.time():06f}"
jobdir_path.mkdir(exist_ok=True, parents=True)
job_runtime_mins = args.runtime_mins if args.runtime_hours is None else args.runtime_hours * 60


def get_runtime_format(runtime_mins):
    days = runtime_mins // (60 * 24)
    hours = runtime_mins // 60
    minutes = runtime_mins % 60
    return f"{days:02d}::{hours:02d}:{minutes:02d}:00"


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
srun --gpus={args.num_gpus} --mem-per-gpu={args.mem_per_gpu}G --cpus-per-gpu={args.cpus_per_gpu} --time={get_runtime_format(job_runtime_mins)} --exclude=kd-2080ti-2.grasp.maas --job-name={args.job_name} --qos={args.qos} --container-mounts=../../datasets/:/efs/,`pwd`:/project --container-image={docker_image_path} bash command.sh
"""
    with open(srun_path, "w") as f:
        f.write(srun_file_content)


def make_screen():
    screen_path = jobdir_path / f"runme.sh"
    screen_file_content = f"""#!/bin/bash
screen -L -Logfile {jobdir_path}/stdout.log -dmS {args.job_name} bash {jobdir_path}/srun.sh
"""
    with open(screen_path, "w") as f:
        f.write(screen_file_content)


make_command_file(args.command)
make_srun()
make_screen()
if not args.dry_run:
    run_cmd(f"bash {jobdir_path}/runme.sh")

print(f"Config files written to {jobdir_path.absolute()}")
