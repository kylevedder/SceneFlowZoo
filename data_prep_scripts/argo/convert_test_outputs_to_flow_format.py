import os

# Set Open MP's num threads to 1
os.environ["OMP_NUM_THREADS"] = "1"

from bucketed_scene_flow_eval.utils import load_npy, save_npz
from pathlib import Path
import multiprocessing
import tqdm

import argparse

parser = argparse.ArgumentParser()
# Path to the input folder
parser.add_argument("input_folder", type=Path, help="Path to the input folder")
# Path to the output folder
parser.add_argument("output_folder", type=Path, help="Path to the output folder")
# Number of CPUs to use for parallel processing
parser.add_argument(
    "--num_cpus",
    type=int,
    default=multiprocessing.cpu_count(),
    help="Number of CPUs to use for parallel processing",
)
args = parser.parse_args()

assert args.input_folder.exists(), f"Error: {args.input_folder} does not exist"
assert args.input_folder.is_dir(), f"Error: {args.input_folder} is not a directory"

args.output_folder.mkdir(parents=True, exist_ok=True)


def prepare_jobs():
    sequence_folders = sorted(args.input_folder.glob("*/"))
    return sequence_folders


jobs = prepare_jobs()


def sequence_to_npz_format(data: dict):
    flow = data["est_flow"]
    valid_idxes = data["pc1_flows_valid_idx"]
    delta_time = -1.0
    return {"flow": flow, "valid_idxes": valid_idxes, "delta_time": delta_time}


def process_sequence_folder(folder: Path):
    # Load sequence files
    sequence_files = sorted(folder.glob("*.npy"))
    assert len(sequence_files) > 0, f"Error: {folder} has no .npy files"

    for sequence_file in sequence_files:
        # Load the sequence file
        sequence = load_npy(sequence_file, verbose=False).item()
        # Convert the sequence to the npz format
        sequence = sequence_to_npz_format(sequence)
        # Save the sequence to the output folder
        save_npz(
            args.output_folder / folder.name / (sequence_file.stem + ".npz"),
            sequence,
            verbose=False,
        )


# If CPUs <= 1, don't use multiprocessing
if args.num_cpus <= 1:
    for sequence_folder in tqdm.tqdm(jobs):
        process_sequence_folder(sequence_folder)
else:
    # Otherwise, use multiprocessing
    with multiprocessing.Pool(args.num_cpus) as p:
        # Use tqdm to show progress
        for _ in tqdm.tqdm(p.imap_unordered(process_sequence_folder, jobs), total=len(jobs)):
            pass
