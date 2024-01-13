from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from bucketed_scene_flow_eval.datastructures import (
    SE3,
    GroundTruthPointFlow,
    Timestamp,
)

from pointclouds import from_fixed_array


@dataclass
class BucketedSceneFlowItem:
    """
    Class that contains all the data required for computing scene flow of a single sample

    Args:
        dataset_log_id: string of the log id in the dataset
        dataset_idx: the index of the sample in the dataset
        query_timestamp: An int specifying which timestep in the sequence to start the flow query.
        source_pc: Source point cloud for the scene flow problem, shape <N, 3>
        target_pc: Target point cloud for the scene flow problem, shape <M, 3>
        source_pose: SE3 object for the pose at the source
        target_pose: SE3 object for the pose at the target
        gt_flowed_source_pc: The source point cloud with ground truth flow vectors applied, shape <N, 3>
        gt_pc_class_mask: The class ID for each point in the point cloud, shape <N>
        full_percept_pcs_array_stack: Provided for convenience, the full pc instead of just the flow query points, shape <K, PadN, 3>
        full_percept_pose_array_stack: Full pose, shape <K, 4, 4>
        gt_trajectories: Accessed as Dict[ParticleID, ParticleTrajectory] but backed by a numpy array
    """

    dataset_log_id: str
    dataset_idx: int
    query_timestamp: Timestamp
    source_pc: torch.FloatTensor  # (N, 3)
    target_pc: torch.FloatTensor  # (M, 3)
    source_pose: SE3
    target_pose: SE3
    gt_flowed_source_pc: torch.FloatTensor  # (N, 3)
    gt_pc_class_mask: torch.LongTensor  # (N,)

    # Included for completeness, these are the full percepts provided by the
    # dataloader rather than just the scene flow query points.
    # This allows for the data loader to e.g. provide more point clouds for accumulation.
    full_percept_pcs_array_stack: torch.FloatTensor  # (K, PadN, 3)
    full_percept_pose_array_stack: torch.FloatTensor  # (K, 4, 4)
    gt_trajectories: GroundTruthPointFlow

    def to(self, device: str) -> None:
        """
        Copy tensors in this batch to the target device.

        Args:
            device: the string (and optional ordinal) used to construct the device object ex. 'cuda:0'
        """
        self.source_pc = self.source_pc.to(device)
        self.target_pc = self.target_pc.to(device)
        self.full_percept_pcs_array_stack = self.full_percept_pcs_array_stack.to(device)
        self.full_percept_pose_array_stack = self.full_percept_pose_array_stack.to(device)
        self.gt_flowed_source_pc = self.gt_flowed_source_pc.to(device)
        self.gt_pc_class_mask = self.gt_pc_class_mask.to(device)
        return self

    def __post_init__(self):
        assert self.source_pc.shape[1] == 3, f"Source PC has shape {self.source_pc.shape}"
        assert self.target_pc.shape[1] == 3, f"Target PC has shape {self.target_pc.shape}"
        assert (
            self.gt_flowed_source_pc.shape[1] == 3
        ), f"GT flowed source PC has shape {self.gt_flowed_source_pc.shape}"
        assert (
            self.source_pc.shape == self.gt_flowed_source_pc.shape
        ), f"Source PC {self.source_pc.shape} != GT flowed source PC {self.gt_flowed_source_pc.shape}"

        assert (
            self.gt_pc_class_mask.shape[0] == self.gt_flowed_source_pc.shape[0]
        ), f"GT PC class mask {self.gt_pc_class_mask.shape} != GT flowed source PC {self.gt_flowed_source_pc.shape}"

        assert isinstance(self.source_pose, SE3), f"Source pose is not an SE3"
        assert isinstance(self.target_pose, SE3), f"Target pose is not an SE3"

        assert (
            self.full_percept_pcs_array_stack.shape[2] == 3
        ), f"Percept PC has shape {self.full_percept_pcs_array_stack.shape}, but should be (K, PadN, 3)"
        assert (
            self.full_percept_pose_array_stack.shape[0]
            == self.full_percept_pcs_array_stack.shape[0]
        ), f"Full percept and pose stacks have different number of entries"

    def full_percept(self, idx: int) -> Tuple[np.ndarray, SE3]:
        return from_fixed_array(self.full_percept_pcs_array_stack[idx]), SE3.from_array(
            self.full_percept_pose_array_stack[idx]
        )

    def full_percepts(self) -> List[Tuple[np.ndarray, SE3]]:
        return [self.full_percept(idx) for idx in range(len(self.full_percept_pcs_array_stack))]


# @dataclass
# class BucketedSceneFlowBatchOutput():
#     """
#     A standardized set of outputs for Bucketed Scene Flow evaluation.
#     In this dataclass, N is the number of points in pc0, M is for pc1, and all lists have len = batch size.

#     Args:
#         flow: A list of <N, 3> tensors containing the flow for each point.
#         pc0_points_list: A list of <N, 3> tensors containing pc0.
#         pc0_valid_point_indexes: A list of <N> tensors containing a valid mask for the pointcloud pc0.
#         pc1_points_list: A list of <M, 3> tensors containing pc1.
#         pc1_valid_point_indexes: A list of <M> tensors containing a valid mask for the pointcloud pc1.
#         pc0_warped_points_list: An optional list of <N, 3> tensors containing the points of pc0 with the flow vectors added to them.
#         batch_delta_time: An optional float of the amount of time to compute flow for the batch.
#     """
#     flow: List[torch.FloatTensor]
#     pc0_points_list: List[torch.FloatTensor]
#     pc0_valid_point_indexes: List[torch.LongTensor]
#     pc1_points_list: List[torch.FloatTensor]
#     pc1_valid_point_indexes: List[torch.LongTensor]
#     pc0_warped_points_list: Optional[List[torch.FloatTensor]]
#     batch_delta_time: Optional[float]


@dataclass
class BucketedSceneFlowOutputItem:
    """
    A standardized set of outputs for Bucketed Scene Flow evaluation.
    In this dataclass, N is the number of points in pc0, M is for pc1.

    Args:
        flow: A <N, 3> tensor containing the flow for each point.
        pc0_points: A <N, 3> tensor containing pc0.
        pc0_valid_point_indexes: A <N> tensor containing a valid mask for the pointcloud pc0.
        pc1_points: A <M, 3> tensor containing pc1.
        pc1_valid_point_indexes: A <M> tensor containing a valid mask for the pointcloud pc1.
        pc0_warped_points: A <N, 3> tensor containing the points of pc0 with the flow vectors added to them.
    """

    flow: torch.FloatTensor
    pc0_points: torch.FloatTensor
    pc0_valid_point_indexes: torch.LongTensor
    pc1_points: torch.FloatTensor
    pc1_valid_point_indexes: torch.LongTensor
    pc0_warped_points: torch.FloatTensor

    def to(self, device: str) -> None:
        """
        Copy tensors in this batch to the target device.

        Args:
            device: the string (and optional ordinal) used to construct the device object ex. 'cuda:0'
        """
        self.flow = self.flow.to(device)
        self.pc0_points = self.target_pc.to(device)
        self.pc0_valid_point_indexes = self.pc0_valid_point_indexes.to(device)
        self.pc1_points = self.pc1_points.to(device)
        self.pc1_valid_point_indexes = self.pc1_valid_point_indexes.to(device)
        self.pc0_warped_points = self.pc0_warped_points.to(device)
        return self
