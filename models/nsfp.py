import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from dataloaders import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence
from models.embedders import DynamicVoxelizer
from models.nsfp_baseline import NSFPProcessor


class NSFP(nn.Module):
    def __init__(
        self,
        SEQUENCE_LENGTH,
        iterations: int = 5000,
    ) -> None:
        super().__init__()
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH
        assert (
            self.SEQUENCE_LENGTH == 2
        ), "This implementation only supports a sequence length of 2."
        self.nsfp_processor = NSFPProcessor(iterations=iterations)

    def _validate_input(self, batched_sequence: list[BucketedSceneFlowInputSequence]) -> None:
        for sequence in batched_sequence:
            assert (
                len(sequence) == self.SEQUENCE_LENGTH
            ), f"Expected sequence length of {self.SEQUENCE_LENGTH}, but got {len(sequence)}."

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

            warped_full_pc0 = full_p0.clone() + full_flow

            # Everything is in the ego frame of PC1.
            # We must to transform both warped_full_pc0 and full_p0 to the ego frame of PC0

            # To do this, we must go PC1 -> global -> PC0.
            # We get this by composing PC1 ego_to_global with torch.inv(PC0 ego_to_global).
            transform_matrix = pc1_ego_to_global @ torch.inverse(pc0_ego_to_global)
            # Warp both the full point cloud and the warped point cloud to the ego frame of PC0
            # Translation is irrelevant for the flow, so we only use the rotation part of the transform matrix
            warped_full_pc0 = torch.einsum("ij,nj->ni", transform_matrix[:3, :3], warped_full_pc0)
            full_p0 = torch.einsum("ij,nj->ni", transform_matrix[:3, :3], full_p0)

            ego_flow = warped_full_pc0 - full_p0

            batch_output.append(
                BucketedSceneFlowOutputSequence(
                    ego_flows=torch.unsqueeze(ego_flow, 0),
                    valid_flow_mask=torch.unsqueeze(full_p0_mask, 0),
                )
            )

        return batch_output
