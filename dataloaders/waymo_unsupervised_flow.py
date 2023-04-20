from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any
import pandas as pd
from pointclouds import PointCloud, SE3, SE2
import numpy as np
from loader_utils import load_npz

from . import WaymoSupervisedFlowSequenceLoader, WaymoSupervisedFlowSequence


class WaymoUnsupervisedFlowSequence(WaymoSupervisedFlowSequence):

    def __init__(self,
                 sequence_folder: Path,
                 flow_folder: Path,
                 verbose: bool = False):
        super().__init__(sequence_folder, verbose)
        self.flow_folder = Path(flow_folder)
        self.flow_files = sorted(self.flow_folder.glob('*.npz'))
        # We expect the number of flow files to be one fewer than the number
        # of sequence files, with the last sequence file missing a flow
        assert len(self.flow_files) == len(self.sequence_files) - 1, \
            f'Number of flow files {len(self.flow_files)} does not match number of sequence files {len(self.sequence_files)}'

    def __repr__(self) -> str:
        return f'WaymoUnsupervisedFlowSequence with {len(self)} frames'

    def __len__(self):
        return len(self.flow_files)

    def _load_idx(self, idx: int):
        pc, _, labels, pose = super()._load_idx(idx)
        flow_path = self.flow_files[idx]
        flow_dict = dict(load_npz(flow_path, verbose=False))
        flow = flow_dict['flow']
        valid_idxes = flow_dict['valid_idxes']
        return pc, flow, valid_idxes, labels, pose

    def load(self, idx: int, start_idx: int) -> Dict[str, Any]:
        assert idx < len(
            self
        ), f'idx {idx} out of range, len {len(self)} for {self.dataset_dir}'

        idx_pc, idx_flow, idx_valid_idxes, idx_labels, idx_pose = self._load_idx(
            idx)
        _, _, _, _, start_pose = self._load_idx(start_idx)

        relative_pose = start_pose.inverse().compose(idx_pose)
        relative_global_frame_pc = idx_pc.transform(relative_pose)

        return {
            "relative_pc": relative_global_frame_pc,
            "relative_pose": relative_pose,
            "flow": idx_flow,
            "valid_idxes": idx_valid_idxes,
            "log_id": self.sequence_folder.name,
            "log_idx": idx,
        }


class WaymoUnsupervisedFlowSequenceLoader(WaymoSupervisedFlowSequenceLoader):

    def __init__(self,
                 raw_data_path: Path,
                 flow_data_path: Path,
                 verbose: bool = False):
        super().__init__(raw_data_path, verbose=verbose)
        self.flow_data_path = Path(flow_data_path)
        # Load list of flows from flow_data_path
        flow_dir_list = sorted(self.flow_data_path.glob("*/"))
        self.flow_lookup = {e.name: e for e in flow_dir_list}

        # Intersect the flow_lookup keys with log_lookup
        self.log_lookup = {
            k: v
            for k, v in self.log_lookup.items() if k in self.flow_lookup
        }

        # Set of keys that are in both log_lookup and flow_lookup should now be the same
        assert set(self.log_lookup.keys()) == set(
            self.flow_lookup.keys()
        ), f"Log and flow lookup keys do not match. Missing keys: {set(self.log_lookup.keys()) - set(self.flow_lookup.keys())}"

    def load_sequence(self, log_id: str) -> WaymoSupervisedFlowSequence:
        sequence_folder = self.log_lookup[log_id]
        flow_folder = self.flow_lookup[log_id]
        return WaymoUnsupervisedFlowSequence(sequence_folder,
                                             flow_folder,
                                             verbose=self.verbose)
