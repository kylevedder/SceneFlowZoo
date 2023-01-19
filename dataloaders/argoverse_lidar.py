import numpy as np
import pandas as pd
from pathlib import Path
from pointclouds import PointCloud, SE3, SE2
from loader_utils import load_json
from typing import List, Tuple, Dict, Optional

GROUND_HEIGHT_THRESHOLD = 0.4  # 40 centimeters


class ArgoverseSequence():

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        assert self.dataset_dir.is_dir(
        ), f'dataset_dir {dataset_dir} does not exist'
        self.frame_paths = sorted(
            (self.dataset_dir / 'sensors' / 'lidar').glob('*.feather'))
        assert len(
            self.frame_paths) > 0, f'no frames found in {self.dataset_dir}'
        self.frame_infos = pd.read_feather(self.dataset_dir /
                                           'city_SE3_egovehicle.feather')

        self.timestamp_to_file_map = {int(e.stem): e for e in self.frame_paths}
        file_timestamps = set(self.timestamp_to_file_map.keys())

        self.timestamp_to_info_idx_map = {
            int(timestamp): idx
            for idx, timestamp in enumerate(
                self.frame_infos['timestamp_ns'].values)
        }
        info_timestamps = set(self.timestamp_to_info_idx_map.keys())

        self.timestamp_list = sorted(
            file_timestamps.intersection(info_timestamps))
        assert len(self.timestamp_list
                   ) > 0, f'no timestamps found in {self.dataset_dir}'

        self.raster_heightmap, self.global_to_raster_se2, self.global_to_raster_scale = self._load_ground_height_raster(
        )

    def _load_ground_height_raster(self):
        raster_height_paths = list(
            (self.dataset_dir / 'map').glob("*_ground_height_surface____*.npy"))
        assert len(raster_height_paths
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

    def __repr__(self) -> str:
        return f'ArgoverseSequence with {len(self)} frames'

    def __len__(self):
        return len(self.timestamp_list)

    def _load_pc(self, idx) -> PointCloud:
        assert idx < len(self), f'idx {idx} out of range, len {len(self)} for {self.dataset_dir}'
        timestamp = self.timestamp_list[idx]
        frame_path = self.timestamp_to_file_map[timestamp]
        frame_content = pd.read_feather(frame_path)
        xs = frame_content['x'].values
        ys = frame_content['y'].values
        zs = frame_content['z'].values
        points = np.stack([xs, ys, zs], axis=1)
        return PointCloud(points)

    def _load_pose(self, idx) -> SE3:
        assert idx < len(self), f'idx {idx} out of range, len {len(self)} for {self.dataset_dir}'
        timestamp = self.timestamp_list[idx]
        infos_idx = self.timestamp_to_info_idx_map[timestamp]
        frame_info = self.frame_infos.iloc[infos_idx]
        se3 = SE3.from_rot_w_x_y_z_translation_x_y_z(
            frame_info['qw'], frame_info['qx'], frame_info['qy'],
            frame_info['qz'], frame_info['tx_m'], frame_info['ty_m'],
            frame_info['tz_m'])
        return se3

    def load(self, idx, relative_to_idx) -> (PointCloud, SE3):
        assert idx < len(self), f'idx {idx} out of range, len {len(self)} for {self.dataset_dir}'
        pc = self._load_pc(idx)
        start_pose = self._load_pose(relative_to_idx)
        idx_pose = self._load_pose(idx)
        relative_pose = start_pose.inverse().compose(idx_pose)
        absolute_global_frame_pc = pc.transform(idx_pose)
        is_ground_points = self.is_ground_points(absolute_global_frame_pc)
        pc = pc.mask_points(~is_ground_points)
        relative_global_frame_pc = pc.transform(relative_pose)
        return relative_global_frame_pc, relative_pose

    def load_frame_list(self, relative_to_idx) -> List[Tuple[PointCloud, SE3]]:
        return [self.load(idx, relative_to_idx) for idx in range(len(self))]


class ArgoverseSequenceLoader():

    def __init__(self, sequence_dir: Path):
        self.dataset_dir = Path(sequence_dir)
        assert self.dataset_dir.is_dir(
        ), f'dataset_dir {sequence_dir} does not exist'
        self.log_lookup = {e.name: e for e in self.dataset_dir.glob('*/')}

    def get_sequence_ids(self):
        return sorted(self.log_lookup.keys())

    def load_sequence(self, log_id: str) -> ArgoverseSequence:
        assert log_id in self.log_lookup, f'log_id {log_id} does not exist'
        log_dir = self.log_lookup[log_id]
        assert log_dir.is_dir(), f'log_id {log_id} does not exist'
        return ArgoverseSequence(log_dir)


# def load_argoverse(dataset_dir, log_id):
# sdb = SynchronizationDB(dataset_dir, collect_single_log_id=log_id)
