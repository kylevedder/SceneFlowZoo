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
        self.transform = np.array(frame_transform_lst).reshape(4, 4)

    def __repr__(self):
        return f'WaymoFrame({self.name})'

    def get_log_name(self):
        return self.name.split('_frame_')[0]

    def load_point_cloud(self) -> PointCloud:
        pc = load_npz(self.file_path)
        breakpoint()
        return PointCloud.load(self.file_path)


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

    def _load_pc(self, idx) -> PointCloud:
        assert idx < len(
            self
        ), f'idx {idx} out of range, len {len(self)} for {self.sequence_name}'
        frame = self.sequence_metadata_lst[idx][0]
        return frame.load_point_cloud()

    def _load_pose(self, idx) -> SE3:
        assert idx < len(
            self
        ), f'idx {idx} out of range, len {len(self)} for {self.dataset_dir}'
        timestamp = self.timestamp_list[idx]
        infos_idx = self.timestamp_to_info_idx_map[timestamp]
        frame_info = self.frame_infos.iloc[infos_idx]
        se3 = SE3.from_rot_w_x_y_z_translation_x_y_z(
            frame_info['qw'], frame_info['qx'], frame_info['qy'],
            frame_info['qz'], frame_info['tx_m'], frame_info['ty_m'],
            frame_info['tz_m'])
        return se3

    def load(self, idx: int, relative_to_idx: int) -> Dict[str, Any]:
        assert idx < len(
            self
        ), f'idx {idx} out of range, len {len(self)} for {self.dataset_dir}'
        pc = self._load_pc(idx)
        start_pose = self._load_pose(relative_to_idx)
        idx_pose = self._load_pose(idx)
        relative_pose = start_pose.inverse().compose(idx_pose)
        absolute_global_frame_pc = pc.transform(idx_pose)
        is_ground_points = self.is_ground_points(absolute_global_frame_pc)
        pc = pc.mask_points(~is_ground_points)
        relative_global_frame_pc = pc.transform(relative_pose)
        return {
            "relative_pc": relative_global_frame_pc,
            "relative_pose": relative_pose,
            "log_id": self.log_id,
            "log_idx": idx,
        }

    def load_frame_list(self, relative_to_idx) -> List[Tuple[PointCloud, SE3]]:
        return [self.load(idx, relative_to_idx) for idx in range(len(self))]


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

    def _raw_load_sequence(self, log_id: str) -> WaymoRawSequence:
        assert log_id in self.log_lookup, f'log_id {log_id} does not exist'
        log_dir = self.log_lookup[log_id]
        assert log_dir.is_dir(), f'log_id {log_id} does not exist'
        return WaymoRawSequence(log_id, log_dir, verbose=self.verbose)

    def load_sequence(self, log_id: str) -> WaymoRawSequence:
        metadata_lst = self.log_lookup[log_id]
        return WaymoRawSequence(log_id, metadata_lst, verbose=self.verbose)
