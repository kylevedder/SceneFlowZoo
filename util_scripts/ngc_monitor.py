#!/usr/bin/env python3

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import io
import pandas as pd
import paramiko


@dataclass
class Job:
    id: str
    name: str
    status: str
    duration: str


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor and retry NGC batch jobs.")
    parser.add_argument(
        "launch_script", type=Path, help="Path to the launch script on the remote system"
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Maximum number of retry attempts"
    )
    parser.add_argument(
        "--retry-delay", type=int, default=300, help="Delay between retries in seconds"
    )
    parser.add_argument(
        "--ssh-host", type=str, default="nlaptop", help="SSH host name from SSH config"
    )
    parser.add_argument(
        "--dryrun", action="store_true", help="Perform a dry run without actually relaunching jobs"
    )
    return parser.parse_args()


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


def run_remote_command(client: paramiko.SSHClient, command: str) -> tuple[str, str, int]:
    stdin, stdout, stderr = client.exec_command(command)
    exit_status = stdout.channel.recv_exit_status()
    return stdout.read().decode(), stderr.read().decode(), exit_status


def get_job_prefix(client: paramiko.SSHClient, launch_script: Path) -> str:
    """Extract the job prefix from the launch script."""

    # Updated command to match the full job name pattern
    command = (
        f"grep -m 1 'ngc batch run' {launch_script} | "
        "awk '{for(i=1;i<=NF;i++)if($i==\"--name\")print $(i+1)}' | "
        "sed 's/_[^_]*$//' | tr -d '\n'"  # Remove everything after the last underscore
    )

    stdout, stderr, exit_status = run_remote_command(client, command)

    # If there is an error or no output, raise a more informative exception
    if exit_status != 0 or not stdout:
        raise ValueError(
            "Could not find job prefix in launch script. "
            f"Command output: {stdout}, Error: {stderr}"
        )

    return stdout


def get_launch_commands(client: paramiko.SSHClient, launch_script: Path) -> List[str]:
    """Extract all launch commands from the launch script."""
    command = f"grep 'ngc batch run' {launch_script}"
    stdout, stderr, exit_status = run_remote_command(client, command)
    if exit_status != 0:
        raise ValueError("Could not find launch commands in launch script.")
    return stdout.splitlines()


def get_unlaunched_job_commands(launched_jobs: List[Job], launch_commands: List[str]) -> List[str]:
    """Determine which jobs in the launch script haven't been launched yet."""
    launched_job_names = set(job.name for job in launched_jobs)
    return [
        cmd
        for cmd in launch_commands
        if not any(job_name in cmd for job_name in launched_job_names)
    ]


def get_launched_job_status(client: paramiko.SSHClient, job_prefix: str) -> list[Job]:
    command = "ngc batch list --duration=7D --column=name --column=status --column=duration --status STARTING --status RUNNING --status FINISHED_SUCCESS --status QUEUED --status FAILED --format_type csv"
    stdout, stderr, exit_status = run_remote_command(client, command)
    if exit_status != 0:
        raise RuntimeError(f"Failed to get job status. Error: {stderr}")
    df = pd.read_csv(io.StringIO(stdout), header=0)
    # Convert the names of the columns to lowercase
    df = df.rename(columns={col: col.lower() for col in df.columns})
    # Convert to list of Job objects
    jobs = [Job(**row) for _, row in df.iterrows()]
    # Prune failed jobs that have been retried.
    # If there is a newer job that has the same name, remove the older one *but keep the newest one*.
    # This is a simple way to avoid retrying jobs that have already been retried.
    job_names = set()
    pruned_jobs = []
    for job in jobs:
        if job.name not in job_names:
            job_names.add(job.name)
            pruned_jobs.append(job)

    return [job for job in pruned_jobs if job.name.startswith(job_prefix)]


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
        stdout, stderr, exit_status = run_remote_command(client, launch_command)

        if exit_status == 0:
            logging.info("Job launched successfully.")
            break
        else:
            logging.error(f"Failed to launch job. Error: {stderr}")

        if attempt < max_retries - 1:
            logging.info(f"Waiting {retry_delay} seconds before next retry...")
            time.sleep(retry_delay)


def retry_job(
    client: paramiko.SSHClient,
    job: Job,
    launch_commands: list[str],
    max_retries: int,
    retry_delay: int,
    dry_run: bool = False,
) -> None:

    launch_command = [cmd for cmd in launch_commands if job.name in cmd][0]
    if not launch_command:
        logging.error(f"Couldn't find launch command for job {job.name} in the launch script.")
        return

    # replace 16g with 32g for retry
    launch_command = launch_command.replace(".16g.", ".32g.")
    launch_job(client, launch_command, max_retries, retry_delay, dry_run=dry_run)


def monitor_jobs(
    client: paramiko.SSHClient, args: argparse.Namespace, job_prefix: str, dry_run: bool
) -> None:
    check_interval = 30  # Set check interval to 30 seconds
    launch_commands = get_launch_commands(client, args.launch_script)

    while True:
        job_statuses = get_launched_job_status(client, job_prefix)

        unlaunced_jobs = get_unlaunched_job_commands(job_statuses, launch_commands)

        logging.info(f"Found {len(unlaunced_jobs)} unlaunched jobs.")

        # Launch any unlaunched jobs
        for job_command in unlaunced_jobs:
            logging.info(f"Found unlaunched job: {job_command}")
            launch_job(client, job_command, args.max_retries, args.retry_delay, dry_run=dry_run)

        failed_jobs = [job for job in job_statuses if job.status == "FAILED"]

        logging.info(f"Found {len(failed_jobs)} failed jobs.")

        # Retry failed jobs
        for job in failed_jobs:
            logging.info(f"Job {job.name} failed. Attempting to retry...")
            retry_job(
                client,
                job,
                launch_commands,
                args.max_retries,
                args.retry_delay,
                dry_run=dry_run,
            )

        logging.info(f"Sleeping for {check_interval} seconds before next check...")
        time.sleep(check_interval)


def main() -> None:
    setup_logging()
    args = parse_arguments()

    try:
        client = get_ssh_client(args.ssh_host)
        logging.info(f"Connected to SSH host: {args.ssh_host}")

        job_prefix = get_job_prefix(client, args.launch_script)
        logging.info(f"Detected job prefix: {job_prefix}")

        monitor_jobs(client, args, job_prefix, args.dryrun)
    except KeyboardInterrupt:
        logging.info("Script interrupted by user. Exiting...")
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
    finally:
        if "client" in locals():
            client.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
