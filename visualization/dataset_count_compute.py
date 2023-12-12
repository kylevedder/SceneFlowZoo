# Set OMP_NUM_THREADS=1
import os

os.environ["OMP_NUM_THREADS"] = "1"

import torch  # This is required to not have the parallel hang for some reason
import pandas as pd
import open3d as o3d
from dataloaders import ArgoverseSupervisedFlowSequenceLoader, ArgoverseSupervisedFlowSequence, WaymoSupervisedFlowSequenceLoader
from pointclouds import PointCloud, SE3
import numpy as np
import tqdm
import joblib
import multiprocessing
from loader_utils import save_pickle, save_json
import matplotlib.pyplot as plt
# Import color map for plotting
from matplotlib import cm
import matplotlib
from typing import Tuple, List, Callable, Dict, Optional, Iterable, Iterator
from pathlib import Path
from PIL import Image, ImageDraw
import argparse
import subprocess
import cv2
import hashlib

# Take the save_dir as an argument
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',
                    type=Path,
                    default=Path("/efs/argoverse2/val_dataset_stats/"))
parser.add_argument('--dataset_dir',
                    type=Path,
                    default=Path("/efs/argoverse2/val/"))
parser.add_argument('--flow_dir',
                    type=Path,
                    default=Path("/efs/argoverse2/val_sceneflow/"))
parser.add_argument('--num_workers',
                    type=int,
                    default=multiprocessing.cpu_count())
parser.add_argument('--dataset',
                    type=str,
                    default="argo",
                    choices=["argo", "waymo"])
parser.add_argument('--range_limit',
                    type=float,
                    default=None,
                    help="The range limit for the pointclouds in meters")
args = parser.parse_args()

save_dir = args.save_dir
dataset_dir = args.dataset_dir
flow_dir = args.flow_dir
range_limit = args.range_limit

# Make the save directory if it does not exist
save_dir.mkdir(parents=True, exist_ok=True)
# Assert that dataset dir exists
assert dataset_dir.exists(), f"dataset_dir {dataset_dir} does not exist"
# Assert that flow dir exists
assert flow_dir.exists(), f"flow_dir {flow_dir} does not exist"

if range_limit is not None:
    assert range_limit > 0, f"range_limit must be > 0, but got {range_limit}"

speed_bucket_ticks = [
    0, 0.1, 0.5, 1, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0,
    np.inf
]


class RenderedCanvas():

    @staticmethod
    def extents_from_pc(pc: np.ndarray,
                        margin: float) -> Tuple[float, float, float, float]:
        assert pc.ndim == 2, f"pc must be (N, 3) shape, but got {pc.shape}"
        assert pc.shape[1] == 3, f"pc must be (N, 3) shape, but got {pc.shape}"
        min_x, min_y = np.min(pc[:, :2], axis=0) - margin
        max_x, max_y = np.max(pc[:, :2], axis=0) + margin
        return min_x, max_x, min_y, max_y

    def __init__(self,
                 global_pc: np.ndarray,
                 colors: np.ndarray,
                 pose_list: List[SE3],
                 grid_cell_size: Tuple[float, float],
                 margin: float,
                 extents: Optional[Tuple[float, float, float, float]] = None):
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

        if extents is None:
            self.min_x, self.max_x, self.min_y, self.max_y = self.extents_from_pc(
                global_pc, margin)
        else:
            self.min_x, self.max_x, self.min_y, self.max_y = extents

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

    def to_image(self) -> np.ndarray:
        return (self.canvas * 255).astype(np.uint8)

    def to_open_cv_image(self) -> np.ndarray:
        return cv2.cvtColor(self.to_image(), cv2.COLOR_RGB2BGR)

    def save(self, path: Path, verbose=True):
        path = Path(path)
        # Make the parent directory if it does not exist
        path.parent.mkdir(parents=True, exist_ok=True)
        # Remove the file if it already exists
        if path.exists():
            path.unlink()
        # Save the canvas as a PNG
        Image.fromarray(self.to_image()).save(path)
        if verbose:
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

    def render_video_frames(self) -> Iterator[RenderedCanvas]:
        global_pc = np.concatenate(self.pc_list, axis=0)
        extents = RenderedCanvas.extents_from_pc(global_pc, self.margin)
        for frame_pc, frame_category_ids, pose in zip(self.pc_list,
                                                      self.category_id_list,
                                                      self.pose_list):
            frame_colors = self._category_ids_to_rgbs(frame_category_ids)

            # Sort the global pc by z so that the points with larger Z are at the end of the array.
            # This ensure that the points with larger Z are drawn on top of the points with smaller Z.
            # We do arg sort because we want to sort the colors as well
            sorted_order = np.argsort(frame_pc[:, 2])
            frame_pc = frame_pc[sorted_order]
            frame_colors = frame_colors[sorted_order]

            canvas = RenderedCanvas(frame_pc,
                                    frame_colors, [pose],
                                    self.grid_cell_size,
                                    self.margin,
                                    extents=extents)

            yield canvas

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

    def _category_ids_to_rgbs(self,
                              category_ids,
                              color_map_name: str = "gist_rainbow"):
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

        # We need to shatter classes because the color map is continuous and
        # many of the most common classes are close together.
        category_id_set = np.array(self.sequence.category_ids())
        min_id = np.min(category_id_set)
        max_id = np.max(category_id_set)

        # Ensure start at 0
        category_idxes = category_ids - min_id
        assert np.all(
            category_idxes >= 0
        ), f"category_idxes must be >= 0, but got {category_idxes}"
        assert np.all(
            category_idxes < len(category_id_set)
        ), f"category_idxes must be < len(category_ids), but got {category_idxes} against {len(category_id_set)}"

        shattered_id_set = category_id_set.copy()
        # Shuffle the array in place, seeded by the hash of the color map name
        colormap_hash = hashlib.md5(color_map_name.encode())
        seed_from_hash = int.from_bytes(colormap_hash.digest(), "little")
        rng = np.random.default_rng(seed_from_hash)
        rng.shuffle(shattered_id_set)

        # We can index into the shattered id set with the category_idxes to get the shattered ids
        shattered_category_ids = shattered_id_set[category_idxes]
        assert shattered_category_ids.shape == category_ids.shape

        normalized_colormap_entries = (shattered_category_ids -
                                       min_id) / (max_id - min_id)

        color_map = matplotlib.colormaps[color_map_name]
        result = color_map(normalized_colormap_entries)[:, :3]
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


def save_bev_video(bev_renderer: BEVRenderer, video_save_folder: Path):
    video_save_folder = Path(video_save_folder)
    # Make the parent directory if it does not exist
    video_save_folder.mkdir(parents=True, exist_ok=True)

    # Clear out any existing files in the frame format, so they don't get included in the video
    for file in video_save_folder.glob("frame_*.png"):
        file.unlink()

    # Save the canvas as a PNGs and then convert to a video with ffmpeg
    for idx, rendered_canvas in enumerate(bev_renderer.render_video_frames()):
        rendered_canvas.save(video_save_folder / f"frame_{idx:05d}.png", False)

    output_mp4_path = video_save_folder / "bev.mp4"
    if output_mp4_path.exists():
        output_mp4_path.unlink()

    # Convert to a video with ffmpeg
    # Call FFmpeg to create a video from these images
    # Adjust '-framerate' as needed
    command = [
        '/usr/bin/ffmpeg',
        '-framerate',
        '10',  # Set the frame rate
        '-i',
        f'{video_save_folder}/frame_%05d.png',  # Input file pattern
        '-vf',
        'pad=ceil(iw/2)*2:ceil(ih/2)*2',  # Pad to even width and height
        '-c:v',
        'libx264',  # Codec
        '-profile:v',
        'high',
        '-crf',
        '18',  # Quality, lower is better, typical values range from 18 to 24
        '-pix_fmt',
        'yuv420p',  # Pixel format
        '-threads',
        '1',  # Threads
        f'{output_mp4_path}'  # Output file
    ]

    command_str = ' '.join(command)
    # Save the command to a file for debugging
    with open(video_save_folder / "ffmpeg_command.sh", "w") as f:
        f.write(command_str)

    # Run the command
    # print(f"Running command: {' '.join(command)}")
    # result = subprocess.run(command)
    # assert result.returncode == 0, f"FFMPEG failed with return code {result.returncode}"
    print(f"Saved video to {output_mp4_path}")


def save_bev_thumbnail_info(sequence: ArgoverseSupervisedFlowSequence,
                            renderer: BEVRenderer,
                            speed_buckets: SpeedBucketResult,
                            thumbnail_dir: Path):
    thumbnail_dir = Path(thumbnail_dir)
    # Make the parent directory if it does not exist
    thumbnail_dir.mkdir(parents=True, exist_ok=True)

    # Save the counts array
    save_json(thumbnail_dir / "category_count.json",
              speed_buckets.to_named_dict(sequence.category_id_to_name),
              verbose=False)

    canvas, category_name_to_color = renderer.render(
        sequence.category_id_to_name)

    # Save the canvas as a PNG
    canvas.save(thumbnail_dir / "bev.png", verbose=False)

    save_json(thumbnail_dir / "category_color.json",
              category_name_to_color,
              verbose=False)


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

        if range_limit is not None:
            ego_pc = pc.transform(ego_pose.inverse())
            in_range_mask = np.linalg.norm(ego_pc.points, axis=1) < range_limit
            pc = pc.mask_points(in_range_mask)
            flowed_pc = flowed_pc.mask_points(in_range_mask)
            category_ids = category_ids[in_range_mask]

        speed_bucket_result.update(pc, flowed_pc, category_ids)
        bev_renderer.update(pc, ego_pose, category_ids)

    sequence_save_dir = save_dir / f"range_{range_limit}" / f"{sequence.log_id}"
    save_bev_thumbnail_info(sequence, bev_renderer, speed_bucket_result,
                            sequence_save_dir)

    save_bev_video(bev_renderer, sequence_save_dir / "animation")

    return speed_bucket_result


if args.dataset == "argo":
    sequence_loader = ArgoverseSupervisedFlowSequenceLoader(
        dataset_dir, flow_dir)
elif args.dataset == "waymo":
    sequence_loader = WaymoSupervisedFlowSequenceLoader(dataset_dir)
else:
    raise ValueError(f"Unknown dataset {args.dataset}")

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
