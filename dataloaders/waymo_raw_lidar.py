import numpy as np
import pandas as pd
from pathlib import Path
from pointclouds import PointCloud, SE3, SE2
from loader_utils import load_json, load_npz
from typing import List, Tuple, Dict, Optional, Any
import time
from collections import defaultdict

GROUND_HEIGHT_THRESHOLD = 0.4  # 40 centimeters


class WaymoFrame:

    def __init__(self, frame_dir: Path, frame_name: str,
                 frame_transform_lst: List[float]):
        self.name = frame_name

        self.file_path = Path(frame_dir) / frame_name
        assert self.file_path.is_file(
        ), f'file {self.file_path} does not exist'

        assert len(
            frame_transform_lst
        ) == 16, f'Expected 16 elements, got {len(frame_transform_lst)}'
        self.transform = SE3.from_array(
            np.array(frame_transform_lst).reshape(4, 4))

    def __repr__(self) -> str:
        return f'WaymoFrame({self.name})'

    def get_log_name(self) -> str:
        return self.name.split('_frame_')[0]

    def load_frame_data(self) -> Tuple[PointCloud, np.ndarray, np.ndarray]:
        frame_data = dict(load_npz(self.file_path, verbose=False))['frame']
        # - points - [N, 5] matrix which stores the [x, y, z, intensity, elongation] in the frame reference
        # - flows - [N, 4] matrix where each row is the flow for each point in the form [vx, vy, vz, label] in the reference frame
        assert frame_data.shape[
            1] == 9, f'Expected 9 columns, got {frame_data.shape[1]}'
        points = frame_data[:, :3]
        flows = frame_data[:, 5:8]
        # For some reason the flows are -1s for the first entry
        if (flows == -1).all():
            flows = np.zeros_like(flows)
        
        labels = frame_data[:, 8]
        return PointCloud(points), flows, labels

    def load_point_cloud(self) -> PointCloud:
        points, _, _ = self.load_frame_data()
        return points

    def load_flow(self) -> np.ndarray:
        _, flows, _ = self.load_frame_data()
        return flows

    def load_labels(self) -> np.ndarray:
        _, _, labels = self.load_frame_data()
        return labels

    def load_transform(self) -> SE3:
        return self.transform


class WaymoRawSequence():

    def __init__(self,
                 sequence_name: str,
                 sequence_metadata_lst: List[Tuple[WaymoFrame, WaymoFrame]],
                 verbose: bool = False):
        self.sequence_name = sequence_name
        self.sequence_metadata_lst = sequence_metadata_lst
        

        if verbose:
            print(
                f'Loaded {len(self.sequence_metadata_lst)} frames from {self.sequence_name} at timestamp {time.time():.3f}'
            )

    def __repr__(self) -> str:
        return f'ArgoverseSequence with {len(self)} frames'

    def __len__(self):
        return len(self.sequence_metadata_lst)

    def _get_frame(self, idx: int) -> WaymoFrame:
        assert idx < len(
            self
        ), f'idx {idx} out of range, len {len(self)} for {self.sequence_name}'
        frame, _ = self.sequence_metadata_lst[idx]
        return frame
    
    def _load_ground_height_raster(self):
        raster_height_paths = list(
            (self.dataset_dir /
             'map').glob("*_ground_height_surface____*.npy"))
        assert len(
            raster_height_paths
        ) == 1, f'Expected 1 raster, got {len(raster_height_paths)} in path {self.dataset_dir / "map"}'
        raster_height_path = raster_height_paths[0]

        transform_paths = list(
            (self.dataset_dir / 'map').glob("*img_Sim2_city.json"))
        assert len(transform_paths
                   ) == 1, f'Expected 1 transform, got {len(transform_paths)}'
        transform_path = transform_paths[0]

        raster_heightmap = np.load(raster_height_path)
        transform = load_json(transform_path, verbose=False)

        transform_rotation = np.array(transform['R']).reshape(2, 2)
        transform_translation = np.array(transform['t'])
        transform_scale = np.array(transform['s'])

        transform_se2 = SE2(rotation=transform_rotation,
                            translation=transform_translation)

        return raster_heightmap, transform_se2, transform_scale
    
    def get_ground_heights(self, global_point_cloud: PointCloud) -> np.ndarray:
        """Get ground height for each of the xy locations in a point cloud.
        Args:
            point_cloud: Numpy array of shape (k,2) or (k,3) in global coordinates.
        Returns:
            ground_height_values: Numpy array of shape (k,)
        """

        global_points_xy = global_point_cloud.points[:, :2]

        raster_points_xy = self.global_to_raster_se2.transform_point_cloud(
            global_points_xy) * self.global_to_raster_scale

        raster_points_xy = np.round(raster_points_xy).astype(np.int64)

        ground_height_values = np.full((raster_points_xy.shape[0]), np.nan)
        # outside max X
        outside_max_x = (raster_points_xy[:, 0] >=
                         self.raster_heightmap.shape[1]).astype(bool)
        # outside max Y
        outside_max_y = (raster_points_xy[:, 1] >=
                         self.raster_heightmap.shape[0]).astype(bool)
        # outside min X
        outside_min_x = (raster_points_xy[:, 0] < 0).astype(bool)
        # outside min Y
        outside_min_y = (raster_points_xy[:, 1] < 0).astype(bool)
        ind_valid_pts = ~np.logical_or(
            np.logical_or(outside_max_x, outside_max_y),
            np.logical_or(outside_min_x, outside_min_y))

        ground_height_values[ind_valid_pts] = self.raster_heightmap[
            raster_points_xy[ind_valid_pts, 1], raster_points_xy[ind_valid_pts,
                                                                 0]]

        return ground_height_values
    
    def is_ground_points(self, global_point_cloud: PointCloud) -> np.ndarray:
        """Remove ground points from a point cloud.
        Args:
            point_cloud: Numpy array of shape (k,3) in global coordinates.
        Returns:
            ground_removed_point_cloud: Numpy array of shape (k,3) in global coordinates.
        """
        ground_height_values = self.get_ground_heights(global_point_cloud)
        is_ground_boolean_arr = (
            np.absolute(global_point_cloud[:, 2] - ground_height_values) <=
            GROUND_HEIGHT_THRESHOLD) | (
                np.array(global_point_cloud[:, 2] - ground_height_values) < 0)
        return is_ground_boolean_arr

    def load(self, idx: int, relative_to_idx: int) -> Dict[str, Any]:
        assert idx < len(
            self
        ), f'idx {idx} out of range, len {len(self)} for {self.dataset_dir}'

        idx_frame = self._get_frame(idx)
        start_frame = self._get_frame(relative_to_idx)
        pc = idx_frame.load_point_cloud()
        start_pose = start_frame.load_transform()
        idx_pose = idx_frame.load_transform()
        relative_pose = start_pose.inverse().compose(idx_pose)
        absolute_global_frame_pc = pc.transform(idx_pose)
        is_ground_points = self.is_ground_points(absolute_global_frame_pc)
        pc = pc.mask_points(~is_ground_points)
        relative_global_frame_pc = pc.transform(relative_pose)
        return {
            "relative_pc": relative_global_frame_pc,
            "relative_pose": relative_pose,
            "log_id": self.sequence_name,
            "log_idx": idx,
        }

    def load_frame_list(
            self, relative_to_idx: Optional[int]) -> List[Dict[str, Any]]:

        return [
            self.load(idx,
                      relative_to_idx if relative_to_idx is not None else idx)
            for idx in range(len(self))
        ]


class WaymoRawSequenceLoader():

    def __init__(self,
                 sequence_dir: Path,
                 log_subset: Optional[List[str]] = None,
                 verbose: bool = False):
        self.dataset_dir = Path(sequence_dir)
        self.verbose = verbose
        assert self.dataset_dir.is_dir(
        ), f'dataset_dir {sequence_dir} does not exist'

        # Load metadata
        metadata = load_npz(self.dataset_dir / 'metadata')
        look_up_table = metadata['look_up_table']
        #flows_information = metadata['flows_information']

        # Load log lookup
        self.log_lookup = defaultdict(list)
        for (after_frame, before_frame) in look_up_table:
            before_frame = WaymoFrame(self.dataset_dir, *before_frame)
            after_frame = WaymoFrame(self.dataset_dir, *after_frame)
            self.log_lookup[before_frame.get_log_name()].append(
                (before_frame, after_frame))

        # Intersect with log_subset
        if log_subset is not None:
            self.log_lookup = {
                k: v
                for k, v in sorted(self.log_lookup.items())
                if k in set(log_subset)
            }

    def get_sequence_ids(self):
        return sorted(self.log_lookup.keys())

    def load_sequence(self, log_id: str) -> WaymoRawSequence:
        metadata_lst = self.log_lookup[log_id]
        return WaymoRawSequence(log_id, metadata_lst, verbose=self.verbose)
