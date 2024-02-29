import numpy as np
import pandas as pd
import argparse
from typing import List, Tuple, Dict
from pathlib import Path
import shutil
import multiprocessing
import joblib
import tqdm
from loader_utils import load_npz, save_feather

from bucketed_scene_flow_eval.datasets.argoverse2 import (
    ArgoverseNoFlowSequenceLoader,
    ArgoverseNoFlowSequence,
)

parser = argparse.ArgumentParser(description="Convert NPZ files to Feather format.")
parser.add_argument("raw_data_dir", type=Path)
parser.add_argument("npz_dir", type=Path)
parser.add_argument("output_root_dir", type=Path)

args = parser.parse_args()

assert args.raw_data_dir.exists(), f"{args.raw_data_dir} does not exist."
assert args.npz_dir.exists(), f"{args.npz_dir} does not exist."
args.output_root_dir.mkdir(parents=True, exist_ok=True)


def to_feather_dict(is_valid: np.ndarray, flow: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "is_valid": is_valid.astype(bool),
        "flow_tx_m": flow[:, 0].astype(np.float32),
        "flow_ty_m": flow[:, 1].astype(np.float32),
        "flow_tz_m": flow[:, 2].astype(np.float32),
    }


def process_sequence(sequence_id: str):
    sequence = sequence_loader.load_sequence(sequence_id)
    npz_paths = sorted((args.npz_dir / f"{sequence_id}").glob("*.npz"))

    assert len(sequence) - 1 == len(
        npz_paths
    ), f"Expected {len(sequence) - 1} npz files, got {len(npz_paths)}."
    for idx, npz_path in enumerate(npz_paths):
        npz_data = load_npz(npz_path, verbose=False)
        npz_flow = npz_data["flow"]
        npz_valid_idxes = npz_data["valid_idxes"]

        # Only loading this to get the ground mask and valid mask
        item = sequence.load(idx, idx + 1)

        is_valid_no_ground = np.zeros_like(item.in_range_mask, dtype=bool)
        is_valid_no_ground[npz_valid_idxes] = True
        flow_no_ground = np.zeros_like(item.ego_pc)
        flow_no_ground[npz_valid_idxes] = npz_flow

        is_valid_with_ground = np.zeros_like(item.is_ground_points, dtype=bool)
        is_valid_with_ground[~item.is_ground_points] = is_valid_no_ground
        flow_with_ground = np.zeros_like(item.ego_pc_with_ground)
        flow_with_ground[~item.is_ground_points] = flow_no_ground

        save_feather(
            args.output_root_dir / sequence_id / f"{idx:010d}.feather",
            to_feather_dict(is_valid_with_ground, flow_with_ground),
            verbose=False,
        )


sequence_loader = ArgoverseNoFlowSequenceLoader(args.raw_data_dir)
# Parallelize with multiprocessing
joblib.Parallel(n_jobs=-1, verbose=10)(
    joblib.delayed(process_sequence)(sequence_id)
    for sequence_id in sorted(sequence_loader.get_sequence_ids())
)
