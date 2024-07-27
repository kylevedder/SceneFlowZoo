import logging
import paramiko
from dataclasses import dataclass
from pathlib import Path
import time
import io
import pandas as pd


@dataclass
class LauchCommand:
    job_name: str
    command: str


@dataclass
class NGCJob:
    id: int
    name: str
    status: str
    duration: str


@dataclass
class NGCJobState:
    name: str
    jobs: list[NGCJob]

    def most_advanced_state(self) -> str:
        state_ordering = [
            "FINISHED_SUCCESS",
            "RUNNING",
            "STARTING",
            "QUEUED",
            "FAILED",
            "KILLED_BY_USER",
        ]

        for state in state_ordering:
            if any(job.status == state for job in self.jobs):
                return state
        return f"UNKNOWN ({self.jobs[0].status})"

    def kill_duplicate_jobs(self, client: paramiko.SSHClient) -> None:
        # Ensure that there are not multiple jobs running or queued
        running_states = ["FINISHED_SUCCESS", "RUNNING", "QUEUED", "STARTING"]
        running_jobs = [job for job in self.jobs if job.status in running_states]
        if len(running_jobs) <= 1:
            return

        # Kill duplicates, i.e. the not oldest jobs (smallest ID).
        for duplicate_job in running_jobs[:-1]:
            logging.info(f"Killing duplicate job: {duplicate_job}")
            _run_remote_command(client, f"ngc batch kill {duplicate_job.id}")

    def needs_relaunch(self) -> bool:
        """
        Relaunch as long as all of the jobs are FAILED or KILLED_BY_USER or KILLED_BY_SYSTEM.
        """
        failed_states = ["FAILED", "KILLED_BY_USER", "KILLED_BY_SYSTEM"]
        return all(job.status in failed_states for job in self.jobs)

    def get_job_prefix(self) -> str:
        job_prefix = self.name
        if "_" in job_prefix:
            job_prefix = "_".join(job_prefix.split("_")[:-1])
        return job_prefix


@dataclass
class NCGJobCategories:
    unlaunched_jobs: list[LauchCommand]
    failed_jobs: list[LauchCommand]


def _run_remote_command(client: paramiko.SSHClient, command: str) -> tuple[str, str, int]:
    stdin, stdout, stderr = client.exec_command(command)
    exit_status = stdout.channel.recv_exit_status()
    return stdout.read().decode(), stderr.read().decode(), exit_status


def launch_job(
    client: paramiko.SSHClient,
    launch_command: str,
    max_retries: int,
    retry_delay: int,
    dry_run: bool = False,
) -> None:
    if dry_run:
        logging.info(f"Would have launched job with command: {launch_command}")
        return

    for attempt in range(max_retries):
        logging.info(f"Launch attempt {attempt + 1} for job: {launch_command}")
        stdout, stderr, exit_status = _run_remote_command(client, launch_command)

        if exit_status == 0:
            logging.info("Job launched successfully.")
            break
        else:
            logging.error(f"Failed to launch job. Error: {stderr}")

        if attempt < max_retries - 1:
            logging.info(f"Waiting {retry_delay} seconds before next retry...")
            time.sleep(retry_delay)


def get_ssh_client(hostname: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh_config = paramiko.SSHConfig()
    user_config_file = Path.home() / ".ssh" / "config"
    if user_config_file.exists():
        with user_config_file.open() as f:
            ssh_config.parse(f)

    user_config = ssh_config.lookup(hostname)

    connect_kwargs = {
        "hostname": user_config.get("hostname", hostname),
        "port": int(user_config.get("port", 22)),
        "username": user_config.get("user"),
        "look_for_keys": True,
        "allow_agent": True,
    }

    if "identityfile" in user_config:
        connect_kwargs["key_filename"] = user_config["identityfile"]

    logging.info(f"Attempting to connect with parameters: {connect_kwargs}")

    try:
        client.connect(**connect_kwargs)
        return client
    except paramiko.AuthenticationException:
        logging.error("Authentication failed. Please check your SSH key and permissions.")
        raise
    except paramiko.SSHException as ssh_exception:
        logging.error(f"SSH exception occurred: {str(ssh_exception)}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        raise


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_launch_commands(
    client: paramiko.SSHClient, launch_script: Path
) -> tuple[str, dict[str, LauchCommand]]:
    """Extract all launch commands from the launch script."""
    command = f"grep 'ngc batch run' {launch_script}"
    stdout, stderr, exit_status = _run_remote_command(client, command)
    if exit_status != 0:
        raise ValueError("Could not find launch commands in launch script.")
    raw_commands = stdout.splitlines()
    job_name_to_command = {}
    for raw_command in raw_commands:
        # Extract the job name from the command
        job_name = raw_command.split("--name")[1].strip().split(" ")[0].strip()
        job_name_to_command[job_name] = LauchCommand(job_name, raw_command)

    job_prefix = list(job_name_to_command.keys())[0]
    if "_" in job_prefix:
        job_prefix = "_".join(job_prefix.split("_")[:-1])
    return job_prefix, job_name_to_command


def get_job_states(client: paramiko.SSHClient) -> dict[str, NGCJobState]:
    command = "ngc batch list --duration=7D --column=name --column=status --column=duration --format_type csv --begin-time 2024-07-26::22:30:00"
    stdout, stderr, exit_status = _run_remote_command(client, command)
    if exit_status != 0:
        raise RuntimeError(f"Failed to get job status. Error: {stderr}")
    df = pd.read_csv(io.StringIO(stdout), header=0)
    # Convert the names of the columns to lowercase
    df = df.rename(columns={col: col.lower() for col in df.columns})
    # Convert the "id" column to an integer
    df["id"] = df["id"].astype(int)
    ngc_jobs = [NGCJob(**row) for _, row in df.iterrows()]
    # Sort by ID in descending order to get the latest jobs first
    ngc_jobs = sorted(ngc_jobs, key=lambda job: job.id, reverse=True)

    # Group jobs by their prefix
    job_states: dict[str, NGCJobState] = {}
    for job in ngc_jobs:
        job_state = job_states.get(job.name, NGCJobState(job.name, []))
        job_state.jobs.append(job)
        job_states[job.name] = job_state
    return job_states


def categorize_jobs(
    client: paramiko.SSHClient, launch_commands: dict[str, LauchCommand]
) -> NCGJobCategories:
    job_states = get_job_states(client)
    job_categories = NCGJobCategories([], [])
    for job_name, launch_command in launch_commands.items():
        if job_name not in job_states:
            job_categories.unlaunched_jobs.append(launch_command)
        else:
            job_state = job_states[job_name]
            job_state.kill_duplicate_jobs(client)
            if job_state.needs_relaunch():
                job_categories.failed_jobs.append(launch_command)
    return job_categories
