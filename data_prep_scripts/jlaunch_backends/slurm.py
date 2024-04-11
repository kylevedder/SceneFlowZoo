import click
from pathlib import Path
from .shared_utils import run_cmd


@click.command(help="SLURM job submission with configuration options")
@click.argument("command", type=str)
@click.option(
    "--job_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="./job_dir/",
    help="Directory to store job-related files",
)
@click.option("--num_gpus", type=int, default=1, help="Number of GPUs to request")
@click.option("--cpus_per_gpu", type=int, default=2, help="CPUs to allocate per GPU")
@click.option("--mem_per_gpu", type=int, default=12, help="Memory (in GB) to allocate per GPU")
@click.option("--runtime_mins", type=int, default=180, help="Job runtime in minutes")
@click.option(
    "--runtime_hours",
    type=int,
    default=None,
    help="Job runtime in hours (overrides --runtime_mins)",
)
@click.option("--name", type=str, default="slurm", help="Name for the job")
@click.option("--qos", type=str, default="ee-med", help="Quality of service (QoS) for the job")
@click.option("--partition", type=str, default="eaton-compute", help="Partition to target")
@click.option("--dry_run", is_flag=True, help="Do not submit the job, only print the configuration")
@click.option(
    "--blacklist_substring",
    type=str,
    default=None,
    help="Filter out nodes containing this substring",
)
def backend_slurm(
    command: str,
    job_dir: Path,
    num_gpus: int,
    cpus_per_gpu: int,
    mem_per_gpu: int,
    runtime_mins: int,
    runtime_hours: int,
    name: str,
    qos: str,
    partition: str,
    dry_run: bool,
    blacklist_substring: str,
):
    job_dir = Path(job_dir).absolute()
    num_prior_jobs = len(list([e for e in job_dir.iterdir() if e.is_dir()]))
    jobdir_path = job_dir / f"{num_prior_jobs:06d}"
    jobdir_path.mkdir(exist_ok=True, parents=True)
    job_runtime_mins = runtime_mins if runtime_hours is None else runtime_hours * 60

    node_blacklist = get_node_blacklist(blacklist_substring)
    make_command_file(jobdir_path, command)
    make_sbatch(
        jobdir_path,
        name,
        qos,
        partition,
        job_runtime_mins,
        num_gpus,
        mem_per_gpu,
        cpus_per_gpu,
        node_blacklist,
    )
    if dry_run:
        return
    print("RUN THIS COMMAND TO SUBMIT THE JOB:")
    print("|")
    print("|")
    print("|")
    print("|")
    print("|")
    print("V")
    print(f"sbatch {jobdir_path}/sbatch.bash")
    print("^")
    print("|")
    print("|")
    print("|")
    print("|")
    print("|")

    print(f"Config files written to {jobdir_path.absolute()}")


def load_available_nodes():
    res = run_cmd("sinfo --Node | awk '{print $1}' | tail +2", return_stdout=True)
    available_nodes = res.split("\n")
    return [e.strip() for e in available_nodes]


def get_node_blacklist(blacklist_substring: str):  # Type hint added
    if blacklist_substring is None:
        return []

    node_list = load_available_nodes()
    node_blacklist_list = []
    if "," in blacklist_substring:
        node_blacklist_list = [e.strip() for e in blacklist_substring.split(",")]
    else:
        node_blacklist_list = [blacklist_substring]

    print(f"Blacklisting nodes with substrings {node_blacklist_list}")
    print(f"Available nodes: {node_list}")

    def is_blacklisted(node):
        for blacklist_str in node_blacklist_list:
            if blacklist_str in node:
                return True
        return False

    node_blacklist = [node for node in node_list if is_blacklisted(node)]
    print(f"Blacklisted nodes: {node_blacklist}")
    return node_blacklist


def get_runtime_format(runtime_mins: int):
    hours = runtime_mins // 60
    minutes = runtime_mins % 60
    return f"{hours:02d}:{minutes:02d}:00"


def make_command_file(jobdir_path: Path, command: str):
    command_path = jobdir_path / f"command.sh"
    command_file_content = f"""#!/bin/bash
{command}
"""
    with open(command_path, "w") as f:
        f.write(command_file_content)


def make_sbatch(
    jobdir_path: Path,
    job_name: str,
    qos: str,
    partition: str,
    job_runtime_mins: int,
    num_gpus: int,
    mem_per_gpu: int,
    cpus_per_gpu: int,
    node_blacklist: list[str],
):
    current_working_dir = Path.cwd().absolute()
    sbatch_path = jobdir_path / f"sbatch.bash"
    docker_image_path = Path("kylevedder_offline_sceneflow_latest.sqsh").absolute()
    assert (
        docker_image_path.is_file()
    ), f"Docker image {docker_image_path} squash file does not exist"
    sbatch_file_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --qos={qos}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --output={jobdir_path}/job.out
#SBATCH --error={jobdir_path}/job.err
#SBATCH --time={get_runtime_format(job_runtime_mins)}
#SBATCH --gpus={num_gpus}
#SBATCH --mem-per-gpu={mem_per_gpu}G
#SBATCH --cpus-per-gpu={cpus_per_gpu}
#SBATCH --exclude={','.join(node_blacklist)}
#SBATCH --container-mounts=../../datasets/:/efs/,{current_working_dir}:/project
#SBATCH --container-image={docker_image_path}

bash {jobdir_path}/command.sh && echo 'done' > {jobdir_path}/job.done
"""
    with open(sbatch_path, "w") as f:
        f.write(sbatch_file_content)
