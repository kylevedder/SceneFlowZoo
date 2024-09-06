#!/usr/bin/env python3
# Add current directory to path
import sys

sys.path.append(".")

import argparse
import logging
import sys
import time
from pathlib import Path
import paramiko

from util_scripts.ngc_utils.utils import (
    get_ssh_client,
    setup_logging,
    get_launch_commands,
    categorize_jobs,
    launch_job,
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


def monitor_loop(
    client: paramiko.SSHClient,
    launch_script: Path,
    max_retries: int,
    retry_delay: int,
    dry_run: bool,
) -> None:
    check_interval = 30  # Set check interval to 30 seconds
    job_prefix, launch_commands = get_launch_commands(client, launch_script)
    logging.info(f"Found {len(launch_commands)} launch commands; named {job_prefix}")

    while True:
        categorized_jobs = categorize_jobs(client, launch_commands)

        logging.info(f"For job prefix {job_prefix}:")
        logging.info(f"Found {len(categorized_jobs.unlaunched_jobs)} unlaunched jobs.")
        logging.info(f"Found {len(categorized_jobs.failed_jobs)} failed jobs.")

        # Launch any unlaunched jobs
        for job_command in categorized_jobs.unlaunched_jobs:
            logging.info(f"Launching job: {job_command.job_name}")
            launch_job(client, job_command.command, max_retries, retry_delay, dry_run=dry_run)

        for job in categorized_jobs.failed_jobs:
            logging.info(f"Job {job.job_name} failed. Attempting to retry...")
            if ".16g" in job.command:
                logging.info(f"Retrying {job.job_name} with 32GB memory...")
                job.command = job.command.replace(".16g.", ".32g.")
            launch_job(client, job.command, max_retries, retry_delay, dry_run=dry_run)

        logging.info(f"Sleeping for {check_interval} seconds before next check...")
        time.sleep(check_interval)


def main() -> None:
    setup_logging()
    args = parse_arguments()

    try:
        client = get_ssh_client(args.ssh_host)
        logging.info(f"Connected to SSH host: {args.ssh_host}")

        monitor_loop(client, args.launch_script, args.max_retries, args.retry_delay, args.dryrun)
    except KeyboardInterrupt:
        logging.info("Script interrupted by user. Exiting...")
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
    finally:
        if "client" in locals():
            if client is not None:
                client.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
