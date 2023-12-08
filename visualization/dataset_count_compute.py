# Set OMP_NUM_THREADS=1
import os

os.environ["OMP_NUM_THREADS"] = "1"

import torch
import pandas as pd
import open3d as o3d
from dataloaders import ArgoverseSupervisedFlowSequenceLoader, ArgoverseSupervisedFlowSequence
from pointclouds import PointCloud, SE3
import numpy as np
import tqdm
import joblib
import multiprocessing
from loader_utils import save_pickle, save_json
import matplotlib.pyplot as plt
# Import color map for plotting
from matplotlib import cm
from typing import Tuple
from pathlib import Path
from PIL import Image
import argparse

# Take the save_dir as an argument
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',
                    type=Path,
                    default=Path("dataset_count_compute_save_dir/"))
parser.add_argument('--dataset_dir',
                    type=Path,
                    default=Path("/efs/argoverse2/val/"))
parser.add_argument('--flow_dir',
                    type=Path,
                    default=Path("/efs/argoverse2/val_sceneflow/"))
parser.add_argument('--num_workers',
                    type=int,
                    default=multiprocessing.cpu_count())
args = parser.parse_args()

save_dir = args.save_dir
dataset_dir = args.dataset_dir
flow_dir = args.flow_dir

# Make the save directory if it does not exist
save_dir.mkdir(parents=True, exist_ok=True)
# Assert that dataset dir exists
assert dataset_dir.exists(), f"dataset_dir {dataset_dir} does not exist"
# Assert that flow dir exists
assert flow_dir.exists(), f"flow_dir {flow_dir} does not exist"

CATEGORY_NAME_TO_ID = {
    "ANIMAL": 0,
    "ARTICULATED_BUS": 1,
    "BICYCLE": 2,
    "BICYCLIST": 3,
    "BOLLARD": 4,
    "BOX_TRUCK": 5,
    "BUS": 6,
    "CONSTRUCTION_BARREL": 7,
    "CONSTRUCTION_CONE": 8,
    "DOG": 9,
    "LARGE_VEHICLE": 10,
    "MESSAGE_BOARD_TRAILER": 11,
    "MOBILE_PEDESTRIAN_CROSSING_SIGN": 12,
    "MOTORCYCLE": 13,
    "MOTORCYCLIST": 14,
    "OFFICIAL_SIGNALER": 15,
    "PEDESTRIAN": 16,
    "RAILED_VEHICLE": 17,
    "REGULAR_VEHICLE": 18,
    "SCHOOL_BUS": 19,
    "SIGN": 20,
    "STOP_SIGN": 21,
    "STROLLER": 22,
    "TRAFFIC_LIGHT_TRAILER": 23,
    "TRUCK": 24,
    "TRUCK_CAB": 25,
    "VEHICULAR_TRAILER": 26,
    "WHEELCHAIR": 27,
    "WHEELED_DEVICE": 28,
    "WHEELED_RIDER": 29,
    "BACKGROUND": -1
}

CATEGORY_ID_TO_NAME = {v: k for k, v in CATEGORY_NAME_TO_ID.items()}

CATEGORY_ID_TO_IDX = {
    v: idx
    for idx, v in enumerate(sorted(CATEGORY_NAME_TO_ID.values()))
}
CATEGORY_IDX_TO_ID = {v: k for k, v in CATEGORY_ID_TO_IDX.items()}

speed_bucket_ticks = [
    0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5,
    1.6, 1.7, 1.8, 1.9, 2.0, np.inf
]


def get_speed_bucket_ranges():
    return list(zip(speed_bucket_ticks, speed_bucket_ticks[1:]))


def category_ids_to_rgbs(category_ids):
    """Convert a category ID to an RGB color for plotting"""
    assert category_ids is not None, f"category_ids must not be None"
    # Category ids have to be a numpy array for indexing
    assert isinstance(
        category_ids, np.ndarray
    ), f"category_ids must be a numpy array, but got {type(category_ids)}"
    # Category ids have to be (N, ) shape
    assert category_ids.ndim == 1, f"category_ids must be (N, ) shape, but got {category_ids.shape}"

    is_background_mask = (category_ids == CATEGORY_NAME_TO_ID["BACKGROUND"])

    # We need to shift the category ID by 1 because the background class is -1, and then we need to normalize it
    index = category_ids + 1
    # Normalize the index
    index = index / len(CATEGORY_ID_TO_NAME)
    result = cm.tab20(index)[:, :3]
    # Set the background class to black
    result[is_background_mask] = np.array([0.5, 0.5, 0.5])
    return result


class BEVRenderer():

    def __init__(self,
                 global_pc: np.ndarray,
                 global_colors: np.ndarray,
                 grid_cell_size: Tuple[float, float] = (0.05, 0.05),
                 margin: float = 1.0):
        assert global_pc.shape[0] == global_colors.shape[
            0], f"global_pc and global_colors must have the same number of points, but got {global_pc.shape[0]} and {global_colors.shape[0]}"
        assert global_pc.shape[
            1] == 3, f"global_pc must have 3 columns, but got {global_pc.shape[1]}"
        assert global_colors.shape[
            1] == 3, f"global_colors must have 3 columns, but got {global_colors.shape[1]}"

        # Sort the global pc by z so that the points with larger Z are at the end of the array.
        # This ensure that the points with larger Z are drawn on top of the points with smaller Z.
        # We do arg sort because we want to sort the colors as well
        sorted_order = np.argsort(global_pc[:, 2])
        global_pc = global_pc[sorted_order]
        global_colors = global_colors[sorted_order]

        global_pc_xy = global_pc[:, :2]
        self.global_pc_xy = global_pc_xy
        self.global_colors = global_colors

        # Add a small margin to the canvas
        self.min_x, self.min_y = np.min(global_pc_xy, axis=0) - margin
        self.max_x, self.max_y = np.max(global_pc_xy, axis=0) + margin

        # Compute the canvas size
        canvas_size = (
            int((self.max_x - self.min_x) / grid_cell_size[0]),
            int((self.max_y - self.min_y) / grid_cell_size[1]),
        )

        print(f"Canvas size: {canvas_size}")

        self.canvas = self._make_canvas(canvas_size)

        canvas_coords = self._points_xy_to_canvas_coords(global_pc_xy)

        # Draw the points on the canvas
        self.canvas[canvas_coords[:, 0], canvas_coords[:, 1]] = global_colors

    def _make_canvas(self, canvas_size: Tuple[int, int]):
        # White canvas
        canvas = np.ones((canvas_size[0], canvas_size[1], 3), dtype=np.float32)
        return canvas

    def _points_xy_to_canvas_coords(self, points: np.ndarray):
        # Convert points from xy coordinates to canvas coordinates
        # First shift the points to be in the positive quadrant
        points_xy_shifted = points - np.array([self.min_x, self.min_y])
        # Then scale the points to be in the canvas size
        points_xy_scaled = points_xy_shifted / np.array(
            [self.max_x - self.min_x, self.max_y - self.min_y])
        # Then scale the points to be in the canvas size
        points_canvas_coords = points_xy_scaled * np.array(
            [self.canvas.shape[0], self.canvas.shape[1]])
        # Convert them to integers
        points_canvas_coords = points_canvas_coords.astype(np.int32)
        return points_canvas_coords

    def save(self, path: Path):
        path = Path(path)
        # Make the parent directory if it does not exist
        path.parent.mkdir(parents=True, exist_ok=True)
        # Remove the file if it already exists
        if path.exists():
            path.unlink()
        # Save the canvas as a PNG
        Image.fromarray((self.canvas * 255).astype(np.uint8)).save(path)


def visualize_sequence(sequence: ArgoverseSupervisedFlowSequence):
    pc_list = []
    color_list = []
    category_name_color = {name: None for name in CATEGORY_NAME_TO_ID.keys()}

    for idx in range(len(sequence)):
        centered_frame = frame = sequence.load(idx, idx)
        centered_pc = centered_frame['relative_pc']

        is_within_range = np.linalg.norm(
            centered_pc[:, :2], ord=np.inf, axis=1) < 50.0

        frame = sequence.load(idx, 0)
        pc = frame['relative_pc']
        category_ids = frame['pc_classes']
        if category_ids is None:
            continue

        # Mask out points that are not within range
        pc = pc.mask_points(is_within_range)
        category_ids = category_ids[is_within_range]

        category_colors = category_ids_to_rgbs(category_ids)

        pc_list.append(pc.points)
        color_list.append(category_colors)

        unique_category_ids, counts = np.unique(category_ids,
                                                return_counts=True)
        for unique_category_id, count in zip(unique_category_ids, counts):

            category_name = CATEGORY_ID_TO_NAME[unique_category_id]
            matching_category_mask = (category_ids == unique_category_id)
            if np.any(matching_category_mask):
                category_color = category_colors[matching_category_mask][0]
                category_name_color[category_name] = category_color.tolist()

    global_pc = np.concatenate(pc_list, axis=0)
    global_colors = np.concatenate(color_list, axis=0)

    renderer = BEVRenderer(global_pc, global_colors)
    renderer.save(save_dir / f"{sequence.log_id}" / "bev.png")

    # Save the category name to color mapping as a json blob
    save_json(save_dir / f"{sequence.log_id}" / "category_color.json",
              category_name_color)


def count_sequence(sequence: ArgoverseSupervisedFlowSequence):

    speed_bucket_ranges = get_speed_bucket_ranges()
    count_array = np.zeros((len(CATEGORY_ID_TO_IDX), len(speed_bucket_ranges)),
                           dtype=np.int64)

    for idx in range(len(sequence)):
        centered_frame = frame = sequence.load(idx, idx)
        centered_pc = centered_frame['relative_pc']

        is_within_range = np.linalg.norm(
            centered_pc[:, :2], ord=np.inf, axis=1) < 50.0

        frame = sequence.load(idx, 0)
        pc = frame['relative_pc']
        flowed_pc = frame['relative_flowed_pc']
        category_ids = frame['pc_classes']
        if flowed_pc is None:
            continue
        
        # Mask out points that are not within range
        pc = pc.mask_points(is_within_range)
        category_ids = category_ids[is_within_range]
        flowed_pc = flowed_pc.mask_points(is_within_range)

        flow = flowed_pc.points - pc.points
        for category_id in np.unique(category_ids):
            category_idx = CATEGORY_ID_TO_IDX[category_id]
            category_mask = (category_ids == category_id)
            category_flow = flow[category_mask]
            # convert to m/s
            category_speeds = np.linalg.norm(category_flow, axis=1) * 10.0
            category_speed_buckets = np.digitize(category_speeds,
                                                 speed_bucket_ticks) - 1
            for speed_bucket_idx in range(len(speed_bucket_ranges)):
                bucket_mask = (category_speed_buckets == speed_bucket_idx)
                num_points = np.sum(bucket_mask)
                count_array[category_idx, speed_bucket_idx] += num_points

    count_dict = count_array_to_count_dict(count_array)
    # Save the count dict as a json blob
    save_json(save_dir / f"{sequence.log_id}" / "category_count.json",
              count_dict)

    return count_array


def count_array_to_count_dict(count_array: np.ndarray):
    category_name_to_stats = {}
    for category_idx in range(len(CATEGORY_IDX_TO_ID)):
        category_id = CATEGORY_IDX_TO_ID[category_idx]
        category_name = CATEGORY_ID_TO_NAME[category_id]
        category_count_array = count_array[category_idx]
        category_count_dict = dict(
            (k, int(v))
            for k, v in zip(speed_bucket_ticks, category_count_array))
        category_name_to_stats[category_name] = category_count_dict

    return category_name_to_stats


def process_sequence(sequence: ArgoverseSupervisedFlowSequence):
    visualize_sequence(sequence)
    count_array = count_sequence(sequence)
    return count_array


sequence_loader = ArgoverseSupervisedFlowSequenceLoader(dataset_dir, flow_dir)

sequence_ids = sequence_loader.get_sequence_ids()
sequences = [
    sequence_loader.load_sequence(sequence_id) for sequence_id in sequence_ids
]

if args.num_workers <= 1:
    per_sequence_counts_array_lst = [
        process_sequence(sequence)
        for sequence in tqdm.tqdm(sequences, position=1)
    ]
else:
    # process sequences in parallel with joblib
    per_sequence_counts_array_lst = joblib.Parallel(n_jobs=args.num_workers)(
        joblib.delayed(process_sequence)(sequence)
        for sequence in tqdm.tqdm(sequences))
summed_count_array = sum(per_sequence_counts_array_lst)

# Convert the array to a dictionary of class names to array of counts
category_name_to_stats = count_array_to_count_dict(summed_count_array)

# save the counts array
save_pickle(save_dir / 'dataset_count_info.pkl', category_name_to_stats)
save_json(save_dir / 'dataset_count_info.json', category_name_to_stats)
