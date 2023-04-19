from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any
import pandas as pd
from pointclouds import PointCloud, SE3, SE2
import numpy as np

from . import ArgoverseRawSequence


class ArgoverseUnsupervisedFlowSequence(ArgoverseRawSequence):

    def __init__(self, log_id: str, dataset_dir: Path, flow_data_lst: Path):
        super().__init__(log_id, dataset_dir)

        # The flow data does not have a timestamp, so we need to just rely on the order of the files.
        self.flow_data_files = sorted(flow_data_lst.glob('*.npz'))

        assert len(self.timestamp_list) > len(
            self.flow_data_files
        ), f"More flow data files in {flow_data_lst} than pointclouds in {self.dataset_dir};  {len(self.timestamp_list)} vs {len(self.flow_data_files)}"

        # The first len(self.flow_data_files) timestamps have flow data.
        # We keep those timestamps, plus the final timestamp.
        self.timestamp_list = self.timestamp_list[:len(self.flow_data_files) +
                                                  1]

    def _load_flow(self, idx):
        assert idx < len(
            self
        ), f'idx {idx} out of range, len {len(self)} for {self.dataset_dir}'
        # There is no flow information for the last pointcloud in the sequence.
        if idx == len(self) - 1 or idx == -1:
            return None, None

        flow_data_file = self.flow_data_files[idx]
        flow_info = dict(np.load(flow_data_file))
        flow_0_1, valid_idxes = flow_info['flow'], flow_info['valid_idxes']
        return flow_0_1, valid_idxes

    def load(self, idx, relative_to_idx) -> Dict[str, Any]:
        assert idx < len(
            self
        ), f'idx {idx} out of range, len {len(self)} for {self.dataset_dir}'
        raw_pc = self._load_pc(idx)
        flow_0_1, valid_idxes = self._load_flow(idx)
        start_pose = self._load_pose(relative_to_idx)
        idx_pose = self._load_pose(idx)

        relative_pose = start_pose.inverse().compose(idx_pose)

        absolute_global_frame_pc = raw_pc.transform(idx_pose)
        is_ground_points = self.is_ground_points(absolute_global_frame_pc)
        pc = raw_pc.mask_points(~is_ground_points)
        relative_global_frame_pc = pc.transform(relative_pose)
        return {
            "relative_pc": relative_global_frame_pc,
            "relative_pose": relative_pose,
            "flow": flow_0_1,
            "valid_idxes": valid_idxes,
            "log_id": self.log_id,
            "log_idx": idx,
        }


class ArgoverseUnsupervisedFlowSequenceLoader():

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

        # Raw data folders
        raw_data_folders = sorted(self.raw_data_path.glob('*/'))
        self.sequence_id_to_raw_data = {
            folder.stem: folder
            for folder in raw_data_folders
        }
        # Flow data folders
        flow_data_folders = sorted(self.flow_data_path.glob('*/'))
        self.sequence_id_to_flow_data = {
            folder.stem: folder
            for folder in flow_data_folders
        }

        self.sequence_id_list = sorted(
            set(self.sequence_id_to_raw_data.keys()).intersection(
                set(self.sequence_id_to_flow_data.keys())))

        if log_subset is not None:
            self.sequence_id_list = [
                sequence_id for sequence_id in self.sequence_id_list
                if sequence_id in log_subset
            ]

        self.last_loaded_sequence = None
        self.last_loaded_sequence_id = None

    def get_sequence_ids(self):
        return self.sequence_id_list

    def _load_sequence_raw(
            self, sequence_id: str) -> ArgoverseUnsupervisedFlowSequence:
        assert sequence_id in self.sequence_id_to_flow_data, f'sequence_id {sequence_id} does not exist'
        return ArgoverseUnsupervisedFlowSequence(
            sequence_id, self.sequence_id_to_raw_data[sequence_id],
            self.sequence_id_to_flow_data[sequence_id])

    def load_sequence(self,
                      sequence_id: str) -> ArgoverseUnsupervisedFlowSequence:
        # Basic caching mechanism for repeated loads of the same sequence
        if self.last_loaded_sequence_id != sequence_id:
            self.last_loaded_sequence = self._load_sequence_raw(sequence_id)
            self.last_loaded_sequence_id = sequence_id
        return self.last_loaded_sequence
