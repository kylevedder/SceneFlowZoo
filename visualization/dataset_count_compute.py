# Set OMP_NUM_THREADS=1
import os

os.environ["OMP_NUM_THREADS"] = "1"

import torch  # This is required to not have the parallel hang for some reason
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
from typing import Tuple, List, Callable, Dict
from pathlib import Path
from PIL import Image, ImageDraw
import argparse
# Import dataclass
from dataclasses import dataclass

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

speed_bucket_ticks = [
    0, 0.1, 0.5, 1, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0,
    np.inf
]


class RenderedCanvas():

    def __init__(self, global_pc: np.ndarray, colors: np.ndarray,
                 pose_list: List[SE3], grid_cell_size: Tuple[float, float],
                 margin: float):
        assert global_pc.ndim == 2, f"global_pc must be (N, 3) shape, but got {global_pc.shape}"
        assert global_pc.shape[
            1] == 3, f"global_pc must be (N, 3) shape, but got {global_pc.shape}"
        assert colors.ndim == 2, f"colors must be (N, 3) shape, but got {colors.shape}"
        assert colors.shape[
            1] == 3, f"colors must be (N, 3) shape, but got {colors.shape}"
        assert global_pc.shape[0] == colors.shape[
            0], f"global_pc and colors must have the same number of points, but got {global_pc.shape[0]} and {colors.shape[0]}"

        # Check that the colors are between 0 and 1 inclusive

        assert np.all(colors >= 0), f"colors must be between 0 and 1 inclusive"
        assert np.all(colors <= 1), f"colors must be between 0 and 1 inclusive"

        self.grid_cell_size = grid_cell_size
        global_pc_xy = global_pc[:, :2]
        self.min_x, self.min_y = np.min(global_pc_xy, axis=0) - margin
        self.max_x, self.max_y = np.max(global_pc_xy, axis=0) + margin

        canvas_size = (
            int((self.max_x - self.min_x) / self.grid_cell_size[0]),
            int((self.max_y - self.min_y) / self.grid_cell_size[1]),
        )

        self.canvas = np.ones((canvas_size[0], canvas_size[1], 3),
                              dtype=np.float32)
        self._draw_pc(global_pc, colors)
        self._draw_poses(pose_list)

    def _draw_pc(self, global_pc: np.ndarray, colors: np.ndarray):
        canvas_coords = self._points_to_canvas_coords(self.canvas, global_pc)
        self.canvas[canvas_coords[:, 0], canvas_coords[:, 1]] = colors

    def _draw_pose(self,
                   pose: SE3,
                   color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                   dot_radius: int = 3):
        color_np = np.array(color)
        pose_canvas_center_coord = self._points_to_canvas_coords(
            self.canvas, pose.translation[None, :])[0]

        # Create a small PIL image for the dot
        dot_image = Image.new("L", (dot_radius * 2, dot_radius * 2), 0)
        draw = ImageDraw.Draw(dot_image)

        # Draw the circle in the center of this small image
        draw.ellipse([(0, 0), (dot_radius * 2, dot_radius * 2)], fill=1)

        # Convert the dot image to a NumPy array
        dot_array = np.array(dot_image) / 255

        # Calculate the top-left corner for placing the dot on the main canvas
        top_left_y = max(0, pose_canvas_center_coord[0] - dot_radius)
        top_left_x = max(0, pose_canvas_center_coord[1] - dot_radius)

        # Place the mask of the dot on the main canvas
        self.canvas[top_left_y:top_left_y + dot_radius * 2,
                    top_left_x:top_left_x +
                    dot_radius * 2][dot_array > 0] = color_np

    def _draw_poses(self, pose_list: List[SE3]):
        color = np.array([1.0, 0.0, 0.0])
        color_scale_lst = np.linspace(0.1, 0.9, len(pose_list))
        for color_scale, pose in zip(color_scale_lst, pose_list):
            self._draw_pose(pose, color=color * color_scale)

    def _points_to_canvas_coords(self, canvas: np.ndarray, points: np.ndarray):
        assert points.ndim == 2, f"points must be (N, 3) shape, but got {points.shape}"
        assert points.shape[
            1] == 3, f"points must be (N, 3) shape, but got {points.shape}"
        points_xy = points[:, :2]
        # Convert points from xy coordinates to canvas coordinates
        # First shift the points to be in the positive quadrant
        points_xy_shifted = points_xy - np.array([self.min_x, self.min_y])
        # Then scale the points to be in the canvas size
        points_xy_scaled = points_xy_shifted / np.array(
            [self.max_x - self.min_x, self.max_y - self.min_y])
        # Then scale the points to be in the canvas size
        points_canvas_coords = points_xy_scaled * np.array(
            [canvas.shape[0], canvas.shape[1]])
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
        print(f"Saved {path} of shape {self.canvas.shape}")


class BEVRenderer():

    def __init__(self,
                 sequence: ArgoverseSupervisedFlowSequence,
                 grid_cell_size: Tuple[float, float] = (0.05, 0.05),
                 margin: float = 1.0):
        self.sequence = sequence
        self.grid_cell_size = grid_cell_size
        self.margin = margin

        self.pc_list: List[np.ndarray] = []
        self.pose_list: List[SE3] = []
        self.category_id_list: List[np.ndarray] = []

    def update(self, pc: PointCloud, ego_pose: SE3, category_ids: np.ndarray):
        # Ensure that the pointcloud matches the shape of the category ids
        assert category_ids.ndim == 1, f"category_ids must be (N, ) shape, but got {category_ids.shape}"
        assert pc.points.shape[0] == category_ids.shape[
            0], f"pc and category_ids must have the same number of points, but got {pc.points.shape[0]} and {category_ids.shape[0]}"

        self.pc_list.append(pc.points)
        self.pose_list.append(ego_pose)
        self.category_id_list.append(category_ids)

    def render(
        self, category_id_to_name: Callable[[int], str]
    ) -> Tuple[RenderedCanvas, Dict[str, Tuple[float, float, float]]]:

        global_pc = np.concatenate(self.pc_list, axis=0)
        global_category_ids = np.concatenate(self.category_id_list, axis=0)
        global_colors = self._category_ids_to_rgbs(global_category_ids)

        # Sort the global pc by z so that the points with larger Z are at the end of the array.
        # This ensure that the points with larger Z are drawn on top of the points with smaller Z.
        # We do arg sort because we want to sort the colors as well
        sorted_order = np.argsort(global_pc[:, 2])
        global_pc = global_pc[sorted_order]
        global_colors = global_colors[sorted_order]

        canvas = RenderedCanvas(global_pc, global_colors, self.pose_list,
                                self.grid_cell_size, self.margin)

        # Build color mapping for the legend
        unique_category_ids = np.unique(global_category_ids)
        unique_category_id_colors = self._category_ids_to_rgbs(
            unique_category_ids)

        category_name_to_color = {
            category_id_to_name(id): color.tolist()
            for id, color in zip(unique_category_ids,
                                 unique_category_id_colors)
        }

        return canvas, category_name_to_color

    def _category_ids_to_rgbs(self, category_ids):
        """Convert a category ID to an RGB color for plotting"""
        assert category_ids is not None, f"category_ids must not be None"
        # Category ids have to be a numpy array for indexing
        assert isinstance(
            category_ids, np.ndarray
        ), f"category_ids must be a numpy array, but got {type(category_ids)}"
        # Category ids have to be (N, ) shape
        assert category_ids.ndim == 1, f"category_ids must be (N, ) shape, but got {category_ids.shape}"

        is_background_mask = (
            category_ids == self.sequence.category_name_to_id("BACKGROUND"))

        # We need to shift the category ID by 1 because the background class is -1, and then we need to normalize it
        index = category_ids + 1
        # Normalize the index
        index = index / len(self.sequence.category_ids())
        result = cm.tab20(index)[:, :3]
        # Set the background class to black
        result[is_background_mask] = np.array([0.5, 0.5, 0.5])
        return result


class SpeedBucketResult():

    def __init__(self, category_id_list: List[int],
                 speed_bucket_ticks: List[float]):
        self.category_id_list = category_id_list
        self.speed_bucket_ticks = speed_bucket_ticks
        self.count_array = np.zeros(
            (len(self.category_id_list), len(self.speed_bucket_ranges())),
            dtype=np.int64)

        self.category_id_to_category_idx = {
            id: idx
            for idx, id in enumerate(self.category_id_list)
        }

        self.category_idx_to_category_id = {
            idx: id
            for idx, id in enumerate(self.category_id_list)
        }

    def speed_bucket_ranges(self):
        return list(zip(self.speed_bucket_ticks, self.speed_bucket_ticks[1:]))

    def update(self, pc: PointCloud, flowed_pc: PointCloud,
               category_ids: np.ndarray):
        flow = flowed_pc.points - pc.points
        for category_id in np.unique(category_ids):
            category_idx = self.category_id_to_category_idx[category_id]
            category_mask = (category_ids == category_id)
            category_flow = flow[category_mask]
            # convert to m/s
            category_speeds = np.linalg.norm(category_flow, axis=1) * 10.0
            category_speed_buckets = np.digitize(category_speeds,
                                                 self.speed_bucket_ticks) - 1
            for speed_bucket_idx in range(len(self.speed_bucket_ranges())):
                bucket_mask = (category_speed_buckets == speed_bucket_idx)
                num_points = np.sum(bucket_mask)
                self.count_array[category_idx, speed_bucket_idx] += num_points

    def __add__(self, other):
        if isinstance(other, int):
            return self
        assert isinstance(
            other, SpeedBucketResult
        ), f"other must be a SpeedBucketResult, but got {type(other)}"
        assert np.all(
            self.category_id_list == other.category_id_list
        ), f"category_id_list must be the same, but got {self.category_id_list} and {other.category_id_list}"
        assert np.all(
            self.speed_bucket_ticks == other.speed_bucket_ticks
        ), f"speed_bucket_ticks must be the same, but got {self.speed_bucket_ticks} and {other.speed_bucket_ticks}"
        result = SpeedBucketResult(self.category_id_list,
                                   self.speed_bucket_ticks)
        result.count_array = self.count_array + other.count_array
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def to_named_dict(self, category_id_to_name: Callable[[int], str]):
        result = {}
        for category_idx in range(len(self.category_id_list)):
            category_id = self.category_idx_to_category_id[category_idx]
            category_name = category_id_to_name(category_id)
            category_count_array = self.count_array[category_idx]
            category_count_dict = dict(
                (k, int(v))
                for k, v in zip(self.speed_bucket_ticks, category_count_array))
            result[category_name] = category_count_dict

        return result


def clean_process_sequence(
        sequence: ArgoverseSupervisedFlowSequence) -> SpeedBucketResult:
    speed_bucket_result = SpeedBucketResult(sequence.category_ids(),
                                            speed_bucket_ticks)
    bev_renderer = BEVRenderer(sequence)
    for idx in range(len(sequence)):
        frame = sequence.load(idx, 0)
        pc = frame['relative_pc']
        flowed_pc = frame['relative_flowed_pc']
        category_ids = frame['pc_classes']
        ego_pose = frame['relative_pose']
        if flowed_pc is None:
            continue

        speed_bucket_result.update(pc, flowed_pc, category_ids)
        bev_renderer.update(pc, ego_pose, category_ids)

    rendered_canvas, category_name_to_color_map = bev_renderer.render(
        sequence.category_id_to_name)
    rendered_canvas.save(save_dir / f"{sequence.log_id}" / "bev.png")

    save_json(save_dir / f"{sequence.log_id}" / "category_count.json",
              speed_bucket_result.to_named_dict(sequence.category_id_to_name))

    save_json(save_dir / f"{sequence.log_id}" / "category_color.json",
              category_name_to_color_map)

    return speed_bucket_result


sequence_loader = ArgoverseSupervisedFlowSequenceLoader(dataset_dir, flow_dir)
sequence_ids = sequence_loader.get_sequence_ids()
sequences = [
    sequence_loader.load_sequence(sequence_id) for sequence_id in sequence_ids
]

if args.num_workers <= 1:
    speed_bucket_result_lst = [
        clean_process_sequence(sequence)
        for sequence in tqdm.tqdm(sequences, position=1)
    ]
else:
    # process sequences in parallel with joblib
    speed_bucket_result_lst = joblib.Parallel(n_jobs=args.num_workers)(
        joblib.delayed(clean_process_sequence)(sequence)
        for sequence in tqdm.tqdm(sequences))
summed_speed_bucket_result: SpeedBucketResult = sum(speed_bucket_result_lst)

# save the counts array
save_json(
    save_dir / 'dataset_count_info.json',
    summed_speed_bucket_result.to_named_dict(
        sequence_loader.category_id_to_name))
