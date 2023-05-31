import os
# Set OMP env num threads to 1 to avoid deadlock in multithreading
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from loader_utils import load_npz
import multiprocessing as mp
import tqdm
from typing import Dict, Any, Tuple

# Parse arguments from command line for scene flow masks and scene flow outputs
parser = argparse.ArgumentParser()
parser.add_argument("scene_flow_mask_dir",
                    type=Path,
                    help="Path to scene flow mask directory")
parser.add_argument("scene_flow_output_dir",
                    type=Path,
                    help="Path to scene flow output directory")
parser.add_argument("argoverse_dir",
                    type=Path,
                    help="Path to scene flow output directory")
parser.add_argument("output_dir", type=Path, help="Path to output directory")
# Num CPUs to use for multiprocessing
parser.add_argument("--num_cpus",
                    type=int,
                    default=mp.cpu_count(),
                    help="Number of cpus to use for multiprocessing")
args = parser.parse_args()

assert args.scene_flow_mask_dir.is_dir(
), f"{args.scene_flow_mask_dir} is not a directory"
assert args.scene_flow_output_dir.is_dir(
), f"{args.scene_flow_output_dir} is not a directory"
assert args.argoverse_dir.is_dir(), f"{args.argoverse_dir} is not a directory"

args.output_dir.mkdir(parents=True, exist_ok=True)


def load_feather(filepath: Path):
    filepath = Path(filepath)
    assert filepath.exists(), f'{filepath} does not exist'
    return pd.read_feather(filepath)


def load_scene_flow_mask_from_folder(sequence_folder: Path):
    files = sorted(sequence_folder.glob("*.feather"))
    masks = [load_feather(file)['mask'].to_numpy() for file in files]
    return sequence_folder.stem, {
        file.stem: mask
        for file, mask in zip(files, masks)
    }


def load_scene_flow_output_from_folder(sequence_folder: Path):
    files = sorted(sequence_folder.glob("*.npz"))
    outputs = [dict(load_npz(file, verbose=False)) for file in files]
    return sequence_folder.stem, {
        file.stem: output
        for file, output in zip(files, outputs)
    }


def multiprocess_load(folder: Path, worker_fn):
    sequence_folders = sorted(e for e in folder.glob("*") if e.is_dir())
    # sequence_folders = sequence_folders[:5]
    sequence_lookup = {}
    with mp.Pool(processes=args.num_cpus) as pool:
        for k, v in tqdm.tqdm(pool.imap_unordered(worker_fn, sequence_folders),
                              total=len(sequence_folders)):
            sequence_lookup[k] = v
    return sequence_lookup


print("Loading scene flow masks...")
sequence_mask_lookup = multiprocess_load(args.scene_flow_mask_dir,
                                         load_scene_flow_mask_from_folder)
print("Loading scene flow outputs...")
sequence_output_lookup = multiprocess_load(args.scene_flow_output_dir,
                                           load_scene_flow_output_from_folder)
print("Done loading scene flow masks and outputs")

mask_keys = set(sequence_mask_lookup.keys())
output_keys = set(sequence_output_lookup.keys())
assert mask_keys == output_keys, f"Mask keys {mask_keys} != output keys {output_keys}"


def merge_output_and_mask_data(scene_flow_mask_dir: Path, sequence_name: str,
                               timestamp_to_mask: Dict[str, Any],
                               id_to_output: Dict[str, Any]) -> Dict[str, Any]:
    # Access the argoverse lidar files to get the timestamps.
    argoverse_lidar_path = args.argoverse_dir / scene_flow_mask_dir.stem / sequence_name / "sensors" / "lidar"
    assert argoverse_lidar_path.exists(
    ), f"{argoverse_lidar_path} does not exist"
    argoverse_lidar_files = sorted(argoverse_lidar_path.glob("*.feather"))

    # Should cover all frame *pairs*, hence the -1
    assert len(id_to_output) == len(
        argoverse_lidar_files
    ) - 1, f"for {sequence_name}: output length {len(id_to_output)} != lidar length {len(argoverse_lidar_files)} - 1"

    timestamp_to_lidar_file = {e.stem: e for e in argoverse_lidar_files}

    timestamp_to_output = {
        f.stem: v
        for f, v in zip(argoverse_lidar_files, id_to_output.values())
    }

    timestamp_to_masked_output = {}

    # Check that the masks are corresponding to the lidar files.
    for mask_timestamp in timestamp_to_mask:
        lidar_data = load_feather(timestamp_to_lidar_file[mask_timestamp])
        mask_data = timestamp_to_mask[mask_timestamp]
        assert len(lidar_data) == len(
            mask_data
        ), f"for {sequence_name}: lidar data length {len(lidar_data)} != mask data length {len(mask_data)}"
        output_data = timestamp_to_output[mask_timestamp]
        output_data['submission_idxes'] = np.nonzero(mask_data)[0]
        timestamp_to_masked_output[mask_timestamp] = output_data

    return timestamp_to_masked_output


merged_sequences = [
    (sequence_name,
     merge_output_and_mask_data(args.scene_flow_mask_dir, sequence_name,
                                sequence_mask_lookup[sequence_name],
                                sequence_output_lookup[sequence_name]))
    for sequence_name in sorted(sequence_mask_lookup)
]


def save_sequence(input: Tuple[str, Dict[str, Dict[str, Any]]]):
    sequence_name, timestamp_to_masked_output = input
    # print(f"Saving sequence {sequence_name}")
    for timestamp, data in sorted(timestamp_to_masked_output.items()):
        save_path = args.output_dir / sequence_name / f"{timestamp}.feather"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # We size an output array to be the max of the submission idxes and the valid idxes. This way,
        # we can directly index the flow valid_idxes into the scratch array, and then use the submission_idxes
        # to index into the scratch array to get the submission flow.
        max_scratch_array_size = max(max(data['submission_idxes']),
                                     max(data['valid_idxes'])) + 1
        scratch_flow_array = np.zeros((max_scratch_array_size, 3),
                                      dtype=data['flow'].dtype)
        scratch_flow_array[data['valid_idxes']] = data['flow']
        output_flow_array = scratch_flow_array[data["submission_idxes"]]
        flow_magnitudes = np.linalg.norm(output_flow_array, axis=1)
        is_dynamic = flow_magnitudes > 0.05
        output_flow_array = output_flow_array.astype(np.float16)
        df = pd.DataFrame(output_flow_array,
                          columns=['flow_tx_m', 'flow_ty_m', 'flow_tz_m'])
        df['is_dynamic'] = is_dynamic
        df.to_feather(save_path)


with mp.Pool(processes=args.num_cpus) as pool:
    for _ in tqdm.tqdm(pool.imap_unordered(save_sequence, merged_sequences),
                       total=len(merged_sequences)):
        pass
