from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any
import pandas as pd
from pointclouds import PointCloud, SE3, SE2
import numpy as np
from loader_utils import load_pickle


class WaymoSupervisedFlowSequence():

    def __init__(self, sequence_folder: Path, verbose: bool = False):
        self.sequence_folder = Path(sequence_folder)
        self.sequence_files = sorted(self.sequence_folder.glob('*.pkl'))
        assert len(self.sequence_files
                   ) > 0, f'no frames found in {self.sequence_folder}'

    def __repr__(self) -> str:
        return f'WaymoRawSequence with {len(self)} frames'

    def __len__(self):
        return len(self.sequence_files)

    def _load_idx(self, idx: int):
        assert idx < len(
            self
        ), f'idx {idx} out of range, len {len(self)} for {self.dataset_dir}'
        pickle_path = self.sequence_files[idx]
        pkl = load_pickle(pickle_path)
        pc = PointCloud(pkl['car_frame_pc'])
        flow = pkl['flow']
        labels = pkl['label']
        pose = SE3.from_array(pkl['pose'])
        return pc, flow, labels, pose

    def cleanup_flow(self, flow):
        flow[np.isnan(flow)] = 0
        flow[np.isinf(flow)] = 0
        flow_speed = np.linalg.norm(flow, axis=1)
        flow_speed[flow_speed > 30] = 0
        return flow

    def load(self, idx: int, start_idx: int) -> Dict[str, Any]:
        assert idx < len(
            self
        ), f'idx {idx} out of range, len {len(self)} for {self.dataset_dir}'

        idx_pc, idx_flow, idx_labels, idx_pose = self._load_idx(idx)
        # Unfortunatly, the flow has some artifacts that we need to clean up. These are very adhoc and will
        # need to be updated if the flow is updated.
        idx_flow = self.cleanup_flow(idx_flow)
        _, _, _, start_pose = self._load_idx(start_idx)

        relative_pose = start_pose.inverse().compose(idx_pose)
        relative_global_frame_pc = idx_pc.transform(relative_pose)
        car_frame_flowed_pc = idx_pc.flow(idx_flow)
        relative_global_frame_flowed_pc = car_frame_flowed_pc.transform(
            relative_pose)

        return {
            "relative_pc": relative_global_frame_pc,
            "relative_pose": relative_pose,
            "relative_flowed_pc": relative_global_frame_flowed_pc,
            "pc_classes": idx_labels,
            "pc_is_ground": (idx_labels == -1),
            "log_id": self.sequence_folder.name,
            "log_idx": idx,
        }

    def load_frame_list(
            self, relative_to_idx: Optional[int]) -> List[Dict[str, Any]]:

        return [
            self.load(idx,
                      relative_to_idx if relative_to_idx is not None else idx)
            for idx in range(len(self))
        ]


class WaymoSupervisedFlowSequenceLoader():

    def __init__(self,
                 sequence_dir: Path,
                 log_subset: Optional[List[str]] = None,
                 verbose: bool = False):
        self.dataset_dir = Path(sequence_dir)
        self.verbose = verbose
        assert self.dataset_dir.is_dir(
        ), f'dataset_dir {sequence_dir} does not exist'

        # Load list of sequences from sequence_dir
        sequence_dir_lst = sorted(self.dataset_dir.glob("*/"))

        self.log_lookup = {e.name: e for e in sequence_dir_lst}

        # Intersect with log_subset
        if log_subset is not None:
            self.log_lookup = {
                k: v
                for k, v in sorted(self.log_lookup.items())
                if k in set(log_subset)
            }

    def get_sequence_ids(self):
        return sorted(self.log_lookup.keys())

    def load_sequence(self, log_id: str) -> WaymoSupervisedFlowSequence:
        sequence_folder = self.log_lookup[log_id]
        return WaymoSupervisedFlowSequence(sequence_folder,
                                           verbose=self.verbose)
