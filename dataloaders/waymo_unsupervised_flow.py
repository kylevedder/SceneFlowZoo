from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any
import pandas as pd
from pointclouds import PointCloud, SE3, SE2
import numpy as np
from loader_utils import load_npz

from . import WaymoRawSequence, WaymoRawSequenceLoader, WaymoFrame


class WaymoUnsupervisedFlowSequence(WaymoRawSequence):

    def __init__(self,
                 sequence_name: str,
                 sequence_metadata_lst: List[Tuple[WaymoFrame, WaymoFrame]],
                 flow_dir: Path,
                 verbose: bool = False):
        super().__init__(sequence_name, sequence_metadata_lst, verbose)
        flow_data_dir = Path(flow_dir) / sequence_name
        self.flow_data_files = sorted(flow_data_dir.glob('*.npz'))

    def __len__(self):
        return min(super().__len__(), len(self.flow_data_files))

    def load_flow(self, idx: int):
        assert idx < len(
            self.flow_data_files
        ), f'idx {idx} out of range for total flow files {len(self.flow_data_files)}; self.len {len(self)}'
        flow_data_file = self.flow_data_files[idx]
        flow = dict(load_npz(flow_data_file, verbose=False))['flow']
        return flow

    def load(self, idx: int, relative_to_idx: int) -> Dict[str, Any]:
        assert idx < len(self), f'idx {idx} out of range, len {len(self)}'
        idx_frame = self._get_frame(idx)
        start_frame = self._get_frame(relative_to_idx)
        pc, _, labels = idx_frame.load_frame_data()
        flow_0_1 = self.load_flow(idx)

        assert pc.points.shape[0] == labels.shape[
            0], f'pc and labels have different number of points, pc: {pc.points.shape}, labels: {labels.shape}'

        start_pose = start_frame.load_transform()
        idx_pose = idx_frame.load_transform()
        relative_pose = start_pose.inverse().compose(idx_pose)
        relative_global_frame_pc = pc.transform(relative_pose)
        return {
            "relative_pc": relative_global_frame_pc,
            "relative_pose": relative_pose,
            "flow": flow_0_1,
            "pc_classes": labels,
            "log_id": self.sequence_name,
            "log_idx": idx,
        }


class WaymoUnsupervisedFlowSequenceLoader(WaymoRawSequenceLoader):

    def __init__(self,
                 raw_data_path: Path,
                 flow_data_path: Path,
                 log_subset: Optional[List[str]] = None,
                 verbose: bool = False):
        super().__init__(raw_data_path, log_subset, verbose)
        self.flow_dir = Path(flow_data_path)

    def load_sequence(self, log_id: str) -> WaymoUnsupervisedFlowSequence:
        metadata_lst = self.log_lookup[log_id]
        return WaymoUnsupervisedFlowSequence(log_id,
                                             metadata_lst,
                                             self.flow_dir,
                                             verbose=self.verbose)
