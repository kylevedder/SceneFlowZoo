import numpy as np
import pandas as pd
import argparse
from typing import List, Tuple, Dict
from pathlib import Path
import shutil
import multiprocessing
import joblib
import tqdm

from bucketed_scene_flow_eval.datasets import construct_dataset
from bucketed_scene_flow_eval.datastructures import QuerySceneSequence, GroundTruthParticleTrajectories, O3DVisualizer

parser = argparse.ArgumentParser(
    description="Convert NPZ files to Feather format.")
parser.add_argument("input_root_dir", type=Path)
parser.add_argument("output_root_dir", type=Path)
args = parser.parse_args()

# We want to iterate over the dataset and save each item as a feather file
# using the dataset index as a name.
dataset = construct_dataset(
    "Argoverse2SceneFlow",
    dict(root_dir=args.input_root_dir,
         use_gt_flow=False,
         with_rgb=False,
         with_ground=True))


def save_feather(entries: Dict[str, np.ndarray], output_path: Path):
    df = pd.DataFrame(entries)
    # Make parent dirs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove the file if it exists
    if output_path.exists():
        output_path.unlink()
    df.to_feather(output_path)


def save_item(entry: Tuple[QuerySceneSequence,
                           GroundTruthParticleTrajectories], idx: int):
    query: QuerySceneSequence = entry[0]
    gt: GroundTruthParticleTrajectories = entry[1]
    # gt.world_points.shape is (N, 2, 3)
    assert gt.world_points.shape[
        1] == 2, f"Expected 2 frames, got {gt.world_points.shape[1]}."
    world_flow = gt.world_points[:, 1] - gt.world_points[:, 0]

    assert gt.is_valid.shape[
        1] == 2, f"Expected 2 frames, got {gt.is_valid.shape[1]}."

    # We want to save the following:
    is_valid = gt.is_valid[:, 0] & gt.is_valid[:, 1]
    flow_tx_m = world_flow[:, 0]
    flow_ty_m = world_flow[:, 1]
    flow_tz_m = world_flow[:, 2]

    save_feather(
        {
            "is_valid": is_valid.astype(bool),
            "flow_tx_m": flow_tx_m.astype(np.float32),
            "flow_ty_m": flow_ty_m.astype(np.float32),
            "flow_tz_m": flow_tz_m.astype(np.float32)
        }, args.output_root_dir / query.scene_sequence.log_id /
        f"{idx:010d}.feather")


for idx, entry in enumerate(tqdm.tqdm(dataset)):
    save_item(entry, idx)
