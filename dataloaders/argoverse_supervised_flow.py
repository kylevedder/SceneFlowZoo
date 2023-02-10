from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import pandas as pd
from pointclouds import PointCloud, SE3, SE2
import numpy as np

from . import ArgoverseRawSequence

CATEGORY_MAP = {
    0: 'ANIMAL',
    1: 'ARTICULATED_BUS',
    2: 'BICYCLE',
    3: 'BICYCLIST',
    4: 'BOLLARD',
    5: 'BOX_TRUCK',
    6: 'BUS',
    7: 'CONSTRUCTION_BARREL',
    8: 'CONSTRUCTION_CONE',
    9: 'DOG',
    10: 'LARGE_VEHICLE',
    11: 'MESSAGE_BOARD_TRAILER',
    12: 'MOBILE_PEDESTRIAN_CROSSING_SIGN',
    13: 'MOTORCYCLE',
    14: 'MOTORCYCLIST',
    15: 'OFFICIAL_SIGNALER',
    16: 'PEDESTRIAN',
    17: 'RAILED_VEHICLE',
    18: 'REGULAR_VEHICLE',
    19: 'SCHOOL_BUS',
    20: 'SIGN',
    21: 'STOP_SIGN',
    22: 'STROLLER',
    23: 'TRAFFIC_LIGHT_TRAILER',
    24: 'TRUCK',
    25: 'TRUCK_CAB',
    26: 'VEHICULAR_TRAILER',
    27: 'WHEELCHAIR',
    28: 'WHEELED_DEVICE',
    29: 'WHEELED_RIDER'
}


class ArgoverseSupervisedFlowSequence(ArgoverseRawSequence):

    def __init__(self, log_id: str, dataset_dir: Path,
                 flow_data_lst: List[Tuple[int, Path]]):
        super().__init__(log_id, dataset_dir)

        # Each flow contains info for the t and t+1 timestamps. This means the last pointcloud in the sequence
        # will not have a corresponding flow.
        self.timestamp_to_flow_map = {
            int(timestamp): flow_data_file
            for timestamp, flow_data_file in flow_data_lst
        }
        # Make sure all of the timestamps in self.timestamp_list *other than the last timestamp* have a corresponding flow.
        assert set(self.timestamp_list[:-1]) == set(
            self.timestamp_to_flow_map.keys(
            )), f'Flow data missing for some timestamps in {self.dataset_dir}'

    def _load_flow(self, idx):
        assert idx < len(
            self
        ), f'idx {idx} out of range, len {len(self)} for {self.dataset_dir}'
        # There is no flow information for the last pointcloud in the sequence.
        if idx == len(self) - 1 or idx == -1:
            return None, None, None, None, None, None
        timestamp = self.timestamp_list[idx]
        flow_data_file = self.timestamp_to_flow_map[timestamp]
        flow_info = dict(np.load(flow_data_file))
        flow_0_1 = flow_info['flow_0_1']
        classes_0 = flow_info['classes_0']
        classes_1 = flow_info['classes_1']
        is_ground0 = flow_info['is_ground_0']
        is_ground1 = flow_info['is_ground_1']
        ego_motion = flow_info['ego_motion']
        return flow_0_1, classes_0, classes_1, is_ground0, is_ground1, ego_motion

    def load(self, idx, relative_to_idx) -> (PointCloud, SE3):
        assert idx < len(
            self
        ), f'idx {idx} out of range, len {len(self)} for {self.dataset_dir}'
        raw_pc = self._load_pc(idx)
        flow_0_1, classes_0, _, is_ground0, _, _ = self._load_flow(idx)
        start_pose = self._load_pose(relative_to_idx)
        idx_pose = self._load_pose(idx)

        relative_pose = start_pose.inverse().compose(idx_pose)

        absolute_global_frame_pc = raw_pc.transform(idx_pose)
        is_ground_points = self.is_ground_points(absolute_global_frame_pc)
        pc = raw_pc.mask_points(~is_ground_points)
        relative_global_frame_pc = pc.transform(relative_pose)
        if flow_0_1 is not None:
            ix_plus_one_pose = self._load_pose(idx + 1)
            relative_pose_plus_one = start_pose.inverse().compose(
                ix_plus_one_pose)
            flowed_pc = raw_pc.flow(flow_0_1)
            flowed_pc = flowed_pc.mask_points(~is_ground_points)
            relative_global_frame_flowed_pc = flowed_pc.transform(
                relative_pose_plus_one)
            classes_0 = classes_0[~is_ground_points]
            is_ground0 = is_ground0[~is_ground_points]
        else:
            relative_global_frame_flowed_pc = None
        return {
            "relative_pc": relative_global_frame_pc,
            "relative_pose": relative_pose,
            "relative_flowed_pc": relative_global_frame_flowed_pc,
            "pc_classes": classes_0,
            "pc_is_ground": is_ground0,
            "log_id": self.log_id,
            "log_idx": idx,
        }


class ArgoverseSupervisedFlowSequenceLoader():

    def __init__(self,
                 raw_data_path: Path,
                 flow_data_path: Path,
                 log_subset: Optional[List[str]] = None):
        self.raw_data_path = Path(raw_data_path)
        self.flow_data_path = Path(flow_data_path)
        assert self.raw_data_path.is_dir(
        ), f'raw_data_path {raw_data_path} does not exist'
        assert self.flow_data_path.is_dir(
        ), f'flow_data_path {flow_data_path} does not exist'

        # Convert folder of flow NPZ files to a lookup table for different sequences
        flow_data_files = sorted(self.flow_data_path.glob('*.npz'))
        self.sequence_id_to_flow_lst = defaultdict(list)
        for flow_data_file in flow_data_files:
            sequence_id, timestamp = flow_data_file.stem.split('_')
            timestamp = int(timestamp)
            self.sequence_id_to_flow_lst[sequence_id].append(
                (timestamp, flow_data_file))

        self.sequence_id_to_raw_data = {}
        for sequence_id in self.sequence_id_to_flow_lst.keys():
            sequence_folder = self.raw_data_path / sequence_id
            assert sequence_folder.is_dir(
            ), f'raw_data_path {raw_data_path} does not exist'
            self.sequence_id_to_raw_data[sequence_id] = sequence_folder

        self.sequence_id_lst = sorted(self.sequence_id_to_flow_lst.keys())
        if log_subset is not None:
            self.sequence_id_lst = [
                sequence_id for sequence_id in self.sequence_id_lst
                if sequence_id in log_subset
            ]

        self.last_loaded_sequence = None
        self.last_loaded_sequence_id = None

    def get_sequence_ids(self):
        return self.sequence_id_lst

    def _load_sequence_raw(self, sequence_id: str) -> ArgoverseSupervisedFlowSequence:
        assert sequence_id in self.sequence_id_to_flow_lst, f'sequence_id {sequence_id} does not exist'
        return ArgoverseSupervisedFlowSequence(sequence_id,
                                     self.sequence_id_to_raw_data[sequence_id],
                                     self.sequence_id_to_flow_lst[sequence_id])

    def load_sequence(self, sequence_id: str) -> ArgoverseSupervisedFlowSequence:
        # Basic caching mechanism for repeated loads of the same sequence
        if self.last_loaded_sequence_id != sequence_id:
            self.last_loaded_sequence = self._load_sequence_raw(sequence_id)
            self.last_loaded_sequence_id = sequence_id
        return self.last_loaded_sequence
