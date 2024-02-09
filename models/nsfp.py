import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from dataloaders import BucketedSceneFlowItem, BucketedSceneFlowOutputItem
from models.embedders import DynamicVoxelizer
from models.nsfp_baseline import NSFPProcessor


class NSFP(nn.Module):
    def __init__(
        self,
        VOXEL_SIZE,
        POINT_CLOUD_RANGE,
        SEQUENCE_LENGTH,
        iterations: int = 5000,
    ) -> None:
        super().__init__()
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH
        self.VOXEL_SIZE = VOXEL_SIZE
        self.POINT_CLOUD_RANGE = POINT_CLOUD_RANGE
        assert (
            self.SEQUENCE_LENGTH == 2
        ), "This implementation only supports a sequence length of 2."
        self.nsfp_processor = NSFPProcessor(iterations=iterations)

    def _range_limit_input(self, pc : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        minx, miny, minz, maxx, maxy, maxz = self.POINT_CLOUD_RANGE
        # Extend the X and Y ranges by an additional voxel size to ensure that the edge points are included
        minx -= self.VOXEL_SIZE[0]
        miny -= self.VOXEL_SIZE[1]
        maxx += self.VOXEL_SIZE[0]
        maxy += self.VOXEL_SIZE[1]

        # Limit the point cloud to the specified range
        in_range_mask = (
            (pc[:, 0] >= minx)
            & (pc[:, 0] <= maxx)
            & (pc[:, 1] >= miny)
            & (pc[:, 1] <= maxy)
            & (pc[:, 2] >= minz)
            & (pc[:, 2] <= maxz)
        )

        in_range_pc = pc[in_range_mask]
        return in_range_pc, in_range_mask

    def _range_limit_input_list(self, pc0s : List[torch.Tensor], pc1s : List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:

        pc0_results = [self._range_limit_input(pc) for pc in pc0s]
        pc1_results = [self._range_limit_input(pc) for pc in pc1s]

        return (
            [pc for pc, _ in pc0_results],
            [mask for _, mask in pc0_results],
            [pc for pc, _ in pc1_results],
            [mask for _, mask in pc1_results],
        )

    def _visualize_result(self, pc0_points: torch.Tensor, warped_pc0_points: torch.Tensor):
        # if pc0_points is torch tensor, convert to numpy
        if isinstance(pc0_points, torch.Tensor):
            pc0_points = pc0_points.cpu().numpy()[0]
        if isinstance(warped_pc0_points, torch.Tensor):
            warped_pc0_points = warped_pc0_points.cpu().numpy()[0]

        import open3d as o3d

        line_set = o3d.geometry.LineSet()
        assert len(pc0_points) == len(
            warped_pc0_points
        ), f"pc and flowed_pc must have same length, but got {len(pc0_pcd)} and {len(warped_pc0_points)}"
        line_set_points = np.concatenate([pc0_points, warped_pc0_points], axis=0)

        pc0_pcd = o3d.geometry.PointCloud()
        pc0_pcd.points = o3d.utility.Vector3dVector(pc0_points)
        warped_pc0_pcd = o3d.geometry.PointCloud()
        warped_pc0_pcd.points = o3d.utility.Vector3dVector(warped_pc0_points)

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_set_points)
        lines = np.array([[i, i + len(pc0_points)] for i in range(len(pc0_points))])
        line_set.lines = o3d.utility.Vector2iVector(lines)

        o3d.visualization.draw_geometries([pc0_pcd, warped_pc0_pcd, line_set])

    def forward(
        self, batched_sequence: List[BucketedSceneFlowItem]
    ) -> List[BucketedSceneFlowOutputItem]:
        """
        Args:
            batched_sequence: A list (len=batch size) of BucketedSceneFlowItems.

        Returns:
            A list (len=batch size) of BucketedSceneFlowOutputItems.
        """
        pc0s = [e.source_pc for e in batched_sequence]
        pc1s = [e.target_pc for e in batched_sequence]
        dataset_idxes = [e.dataset_idx for e in batched_sequence]
        log_ids = [e.dataset_log_id for e in batched_sequence]
        return self._model_forward(pc0s, pc1s, dataset_idxes, log_ids)

    def _process_batch_entry(self, pc0_points, pc1_points) -> Tuple[np.ndarray, float]:
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

        delta_time = after_time - before_time  # How long to do the optimization
        # self._visualize_result(pc0_points, warped_pc0_points)
        flow = warped_pc0_points - pc0_points

        return flow.squeeze(0), delta_time

    def _model_forward(
        self, full_pc0s : List[torch.Tensor], full_pc1s : List[torch.Tensor], dataset_idxes, log_ids
    ) -> List[BucketedSceneFlowOutputItem]:
        (
            pc0_points_lst,
            pc0_valid_point_masks,
            pc1_points_lst,
            pc1_valid_point_masks,
        ) = self._range_limit_input_list(full_pc0s, full_pc1s)

        # process minibatch
        batch_output: List[BucketedSceneFlowOutputItem] = []
        for (
            full_p0,
            full_p1,
            pc0_points,
            pc1_points,
            pc0_point_mask,
            pc1_point_mask,
            dataset_idx,
            log_id,
        ) in zip(
            full_pc0s,
            full_pc1s,
            pc0_points_lst,
            pc1_points_lst,
            pc0_valid_point_masks,
            pc1_valid_point_masks,
            dataset_idxes,
            log_ids,
        ):
            valid_flow, _ = self._process_batch_entry(pc0_points, pc1_points)

            full_flow = torch.zeros_like(full_p0)
            full_flow[pc0_point_mask] = valid_flow

            warped_pc0 = full_p0.clone()
            warped_pc0 += full_flow

            batch_output.append(
                BucketedSceneFlowOutputItem(
                    flow=full_flow,  # type: ignore[arg-type]
                    pc0_points=full_p0,
                    pc0_valid_point_mask=pc0_point_mask,
                    pc0_warped_points=warped_pc0,
                )
            )

        return batch_output
