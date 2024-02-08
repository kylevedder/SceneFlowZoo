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
        flow_save_folder: Path,
    ) -> None:
        super().__init__()
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH
        assert (
            self.SEQUENCE_LENGTH == 2
        ), "This implementation only supports a sequence length of 2."
        self.voxelizer = DynamicVoxelizer(
            voxel_size=VOXEL_SIZE, point_cloud_range=POINT_CLOUD_RANGE
        )
        self.nsfp_processor = NSFPProcessor()
        self.flow_save_folder = Path(flow_save_folder)
        self.flow_save_folder.mkdir(parents=True, exist_ok=True)

    def _voxelize_batched_sequence(self, pc0s, pc1s):
        pc0_voxel_infos_lst = self.voxelizer(pc0s)
        pc1_voxel_infos_lst = self.voxelizer(pc1s)

        pc0_points_lst = [e["points"] for e in pc0_voxel_infos_lst]
        pc0_valid_point_idxes = [e["point_idxes"] for e in pc0_voxel_infos_lst]
        pc1_points_lst = [e["points"] for e in pc1_voxel_infos_lst]
        pc1_valid_point_idxes = [e["point_idxes"] for e in pc1_voxel_infos_lst]

        return pc0_points_lst, pc0_valid_point_idxes, pc1_points_lst, pc1_valid_point_idxes

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

    def _process_batch_entry(
        self, pc0_points, pc1_points, pc0_valid_point_idx, dataset_idx, log_id
    ) -> Tuple[np.ndarray, float]:
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
    
    def _indices_to_mask(self, points, mask_indices):
        mask = torch.zeros((points.shape[0], ), dtype=torch.bool)
        assert mask_indices.max() < len(mask), f"Mask indices go outside of the tensor range. {mask_indices.max()} >= {len(mask)}"
        mask[mask_indices] = True
        return mask

    def _model_forward(
        self, full_pc0s, full_pc1s, dataset_idxes, log_ids
    ) -> List[BucketedSceneFlowOutputItem]:
        (
            pc0_points_lst,
            pc0_valid_point_idxes,
            pc1_points_lst,
            pc1_valid_point_idxes,
        ) = self._voxelize_batched_sequence(full_pc0s, full_pc1s)

        # process minibatch
        batch_output: List[BucketedSceneFlowOutputItem] = []
        for (
            full_p0,
            full_p1,
            pc0_points,
            pc1_points,
            pc0_valid_point_idx,
            pc1_valid_point_idx,
            dataset_idx,
            log_id,
        ) in zip(
            full_pc0s,
            full_pc1s,
            pc0_points_lst,
            pc1_points_lst,
            pc0_valid_point_idxes,
            pc1_valid_point_idxes,
            dataset_idxes,
            log_ids,
        ):
            flow, _ = self._process_batch_entry(
                pc0_points, pc1_points, pc0_valid_point_idx, dataset_idx, log_id
            )

            pc0_point_mask = self._indices_to_mask(full_p0, pc0_valid_point_idx)
            pc1_point_mask = self._indices_to_mask(full_p1, pc1_valid_point_idx)

            warped_pc0 = full_p0.clone()
            warped_pc0[pc0_point_mask] += flow

            batch_output.append(
                BucketedSceneFlowOutputItem(
                    flow=flow,
                    pc0_points=full_p0,
                    pc0_valid_point_mask=pc0_point_mask,
                    pc1_points=full_p1,
                    pc1_valid_point_mask=pc1_point_mask,
                    pc0_warped_points=warped_pc0,
                )
            )

        return batch_output


class NSFPCached(NSFP):
    def __init__(
        self, VOXEL_SIZE, POINT_CLOUD_RANGE, SEQUENCE_LENGTH, flow_save_folder: Path
    ) -> None:
        super().__init__(VOXEL_SIZE, POINT_CLOUD_RANGE, SEQUENCE_LENGTH, flow_save_folder)
        # Implement basic caching to avoid repeated folder reads for the same log.
        self.cached_flow_folder_id = ""
        self.cached_flow_folder_lookup: Dict[int, Path] = {}

    def _setup_folder_cache(self, log_id: str):
        self.cached_flow_folder_id = log_id
        flow_folder = self.flow_save_folder / log_id
        assert flow_folder.is_dir(), f"{flow_folder} does not exist"
        self.cached_flow_folder_lookup = {
            int(e.stem.split("_")[0]): e for e in sorted(flow_folder.glob("*.npz"))
        }

    def _load_result(self, log_id: str, dataset_idx: int):
        if self.cached_flow_folder_id != log_id:
            self._setup_folder_cache(log_id)

        flow_file = self.cached_flow_folder_lookup[dataset_idx]
        print(f"Loading flow from {flow_file}")
        data = dict(np.load(flow_file, allow_pickle=True))
        flow = data["flow"]
        valid_idxes = data["valid_idxes"]
        delta_time = data["delta_time"]
        return flow, valid_idxes, delta_time

    def _process_batch_entry(
        self, pc0_points, pc1_points, pc0_valid_point_idx, dataset_idx, log_id
    ) -> Optional[Tuple[np.ndarray, float]]:
        flow, _, delta_time = self._load_result(log_id, dataset_idx)
        return flow, delta_time
