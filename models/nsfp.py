import time

import torch
import torch.nn as nn

from dataloaders import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence
from models.nsfp_baseline import NSFPProcessor
from .base_model import BaseModel


class NSFP(BaseModel):
    def __init__(
        self,
        sequence_length,
        iterations: int = 5000,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        assert (
            self.sequence_length == 2
        ), "This implementation only supports a sequence length of 2."
        self.nsfp_processor = NSFPProcessor(iterations=iterations)

    def _validate_input(self, batched_sequence: list[BucketedSceneFlowInputSequence]) -> None:
        for sequence in batched_sequence:
            assert (
                len(sequence) == self.sequence_length
            ), f"Expected sequence length of {self.sequence_length}, but got {len(sequence)}."

    def forward(
        self, batched_sequence: list[BucketedSceneFlowInputSequence]
    ) -> list[BucketedSceneFlowOutputSequence]:
        """
        Args:
            batched_sequence: A list (len=batch size) of BucketedSceneFlowItems.

        Returns:
            A list (len=batch size) of BucketedSceneFlowOutputItems.
        """
        self._validate_input(batched_sequence)
        pc0s = [(e.get_full_global_pc(-2), e.get_full_pc_mask(-2)) for e in batched_sequence]
        pc0_transforms = [e.get_pc_transform_matrices(-2) for e in batched_sequence]
        pc1s = [(e.get_full_global_pc(-1), e.get_full_pc_mask(-1)) for e in batched_sequence]
        pc1_transforms = [e.get_pc_transform_matrices(-1) for e in batched_sequence]
        return self._model_forward(pc0s, pc1s, pc0_transforms, pc1_transforms)

    def optimize_single_frame_pair(
        self, pc0_points: torch.Tensor, pc1_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Process a pair of point clouds using NSFP to return flow
        """

        pc0_points = torch.unsqueeze(pc0_points, 0)
        pc1_points = torch.unsqueeze(pc1_points, 0)

        with torch.inference_mode(False):
            with torch.enable_grad():
                pc0_points_new = pc0_points.clone().detach().requires_grad_(True)
                pc1_points_new = pc1_points.clone().detach().requires_grad_(True)

                self.nsfp_processor.train()

                before_time = time.time()
                warped_pc0_points, _ = self.nsfp_processor(
                    pc0_points_new, pc1_points_new, pc1_points_new.device
                )
                after_time = time.time()

        flow = warped_pc0_points - pc0_points
        return flow.squeeze(0)

    def _transform_pc(self, pc: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        """
        Transform an Nx3 point cloud by a 4x4 transformation matrix.
        """

        homogenious_pc = torch.cat((pc, torch.ones((pc.shape[0], 1), device=pc.device)), dim=1)
        return torch.matmul(transform, homogenious_pc.T).T[:, :3]

    def _global_to_ego_flow(
        self,
        global_full_pc0: torch.Tensor,
        global_warped_full_pc0: torch.Tensor,
        pc0_ego_to_global: torch.Tensor,
    ) -> torch.Tensor:

        ego_full_pc0 = self._transform_pc(global_full_pc0, pc0_ego_to_global)
        ego_warped_full_pc0 = self._transform_pc(global_warped_full_pc0, pc0_ego_to_global)

        return ego_warped_full_pc0 - ego_full_pc0

    def _model_forward(
        self,
        full_pc0s: list[tuple[torch.Tensor, torch.Tensor]],
        full_pc1s: list[tuple[torch.Tensor, torch.Tensor]],
        pc0_transforms: list[tuple[torch.Tensor, torch.Tensor]],
        pc1_transforms: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[BucketedSceneFlowOutputSequence]:

        # process minibatch
        batch_output: list[BucketedSceneFlowOutputSequence] = []
        for (
            (full_p0, full_p0_mask),
            (full_p1, full_p1_mask),
            (pc0_sensor_to_ego, pc0_ego_to_global),
            (pc1_sensor_to_ego, pc1_ego_to_global),
        ) in zip(
            full_pc0s,
            full_pc1s,
            pc0_transforms,
            pc1_transforms,
        ):
            masked_p0 = full_p0[full_p0_mask]
            masked_p1 = full_p1[full_p1_mask]
            masked_flow = self.optimize_single_frame_pair(masked_p0, masked_p1)

            full_flow = torch.zeros_like(full_p0)
            full_flow[full_p0_mask] = masked_flow

            ego_flow = self.global_to_ego_flow(full_p0, full_flow, pc0_ego_to_global)

            batch_output.append(
                BucketedSceneFlowOutputSequence(
                    ego_flows=torch.unsqueeze(ego_flow, 0),
                    valid_flow_mask=torch.unsqueeze(full_p0_mask, 0),
                )
            )

        return batch_output
