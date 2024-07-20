#!/usr/bin/env python3

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import io
import pandas as pd
import paramiko


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor NGC batch jobs and report status counts.")
    parser.add_argument(
        "--ssh-host", type=str, default="nlaptop", help="SSH host name from SSH config"
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


def get_job_status(client: paramiko.SSHClient) -> pd.DataFrame:
    command = "ngc batch list --duration=7D --column=name --column=status --format_type csv"
    stdout, stderr, exit_status = run_remote_command(client, command)
    if exit_status != 0:
        raise RuntimeError(f"Failed to get job status. Error: {stderr}")
    df = pd.read_csv(io.StringIO(stdout), header=0)
    # Convert the names of the columns to lowercase
    df = df.rename(columns={col: col.lower() for col in df.columns})
    return df


def group_jobs_by_prefix(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    grouped_jobs = defaultdict(lambda: defaultdict(int))
    latest_jobs = {}  # To keep track of the latest job for each unique name

    # Ensure the dataframe is sorted by job ID in descending order (latest first)
    df = df.sort_values(by="id", ascending=False)

    for _, row in df.iterrows():
        name = row["name"]
        status = row["status"]

        # If we've already seen this job name, skip this older instance
        if name in latest_jobs:
            continue

        latest_jobs[name] = row["id"]  # Mark this as the latest job for this name

        # Split on the last underscore to get the prefix
        prefix = "_".join(name.split("_")[:-1])

        grouped_jobs[prefix][status] += 1
        grouped_jobs[prefix]["TOTAL"] += 1

    return grouped_jobs


def print_job_status_report(grouped_jobs: Dict[str, Dict[str, int]]) -> None:
    for prefix, status_counts in grouped_jobs.items():
        print(f"\nJob Prefix: {prefix}")
        print("-" * 20)
        for status, count in status_counts.items():
            if status != "TOTAL":
                print(f"{status}: {count}")
        print(f"TOTAL: {status_counts['TOTAL']}")
        print()


def main() -> None:
    setup_logging()
    args = parse_arguments()

    try:
        client = get_ssh_client(args.ssh_host)
        logging.info(f"Connected to SSH host: {args.ssh_host}")

        job_status_df = get_job_status(client)
        grouped_jobs = group_jobs_by_prefix(job_status_df)
        print_job_status_report(grouped_jobs)

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
