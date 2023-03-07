from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any
import pandas as pd
from pointclouds import PointCloud, SE3, SE2
import numpy as np

from . import WaymoRawSequence, WaymoRawSequenceLoader


class WaymoSupervisedFlowSequence(WaymoRawSequence):

    def load(self, idx: int, relative_to_idx: int) -> Dict[str, Any]:
        assert idx < len(
            self
        ), f'idx {idx} out of range, len {len(self)} for {self.dataset_dir}'
        idx_frame = self._get_frame(idx)
        relative_to_frame = self._get_frame(relative_to_idx)
        pc, flow, labels = idx_frame.load_frame_data()

        assert pc.points.shape[0] == flow.shape[
            0], f'pc and flow have different number of points, pc: {pc.points.shape[0]}, flow: {flow.shape[0]}'
        assert pc.points.shape[0] == labels.shape[
            0], f'pc and labels have different number of points, pc: {pc.points.shape[0]}, labels: {labels.shape[0]}'

        flowed_pc = pc.flow(flow)

        start_pose = relative_to_frame.load_transform()
        idx_pose = relative_to_frame.load_transform()
        relative_pose = start_pose.inverse().compose(idx_pose)
        # absolute_global_frame_pc = pc.transform(idx_pose)
        # is_ground_points = self.is_ground_points(absolute_global_frame_pc)
        # pc = pc.mask_points(~is_ground_points)
        relative_global_frame_pc = pc.transform(relative_pose)
        relative_global_frame_flowed_pc = flowed_pc.transform(relative_pose)
        #     return {
        #         "relative_pc": relative_global_frame_pc,
        #         "relative_pose": relative_pose,
        #         "relative_flowed_pc": relative_global_frame_flowed_pc,
        #         "pc_classes": classes_0,
        #         "pc_is_ground": is_ground0,
        #         "log_id": self.log_id,
        #         "log_idx": idx,
        #     }
        return {
            "relative_pc": relative_global_frame_pc,
            "relative_pose": relative_pose,
            "relative_flowed_pc": relative_global_frame_flowed_pc,
            "pc_classes": labels,
            "log_id": self.sequence_name,
            "log_idx": idx,
        }


class WaymoSupervisedFlowSequenceLoader(WaymoRawSequenceLoader):

    def _raw_load_sequence(self, log_id: str) -> WaymoSupervisedFlowSequence:
        assert log_id in self.log_lookup, f'log_id {log_id} does not exist'
        log_dir = self.log_lookup[log_id]
        assert log_dir.is_dir(), f'log_id {log_id} does not exist'
        return WaymoSupervisedFlowSequence(log_id,
                                           log_dir,
                                           verbose=self.verbose)

    def load_sequence(self, log_id: str) -> WaymoSupervisedFlowSequence:
        metadata_lst = self.log_lookup[log_id]
        return WaymoSupervisedFlowSequence(log_id,
                                           metadata_lst,
                                           verbose=self.verbose)
