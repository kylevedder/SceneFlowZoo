#!/usr/bin/env python3
# Add current directory to path
import sys

sys.path.append(".")


import argparse
import sys
import paramiko


from util_scripts.ngc_utils.utils import (
    get_ssh_client,
    NGCJobState,
    get_job_states,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor NGC batch jobs and report status counts.")
    parser.add_argument(
        "--ssh-host", type=str, default="nlaptop", help="SSH host name from SSH config"
    )
    return parser.parse_args()


def print_jobs(
    client: paramiko.SSHClient,
) -> None:
    print("Retrieving job states...")
    job_states = get_job_states(client)
    print("Retrieved job states.")
    prefix_lookup: dict[str, list[NGCJobState]] = {}

    for job in job_states.values():
        prefix = job.get_job_prefix()
        prefix_jobs = prefix_lookup.get(prefix, [])
        prefix_jobs.append(job)
        prefix_lookup[prefix] = prefix_jobs

    for prefix, jobs in prefix_lookup.items():
        most_advanced_states = [job.most_advanced_state() for job in jobs]
        state_counts = {
            state: most_advanced_states.count(state) for state in set(most_advanced_states)
        }
        print("========================================")
        print(f"Job Prefix: {prefix}")
        print("========================================")
        for state, count in state_counts.items():
            print(f"\t{state}: {count}")


def main() -> None:
    args = parse_arguments()

    try:
        client = get_ssh_client(args.ssh_host)

        print_jobs(client)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if "client" in locals():
            client.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
