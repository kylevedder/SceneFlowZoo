import torch
import torch.nn as nn
from mmcv.ops import Voxelization
from typing import List, Tuple


class HardVoxelizer(nn.Module):

    def __init__(self, voxel_size, point_cloud_range,
                 max_points_per_voxel: int):
        super().__init__()
        assert max_points_per_voxel > 0, f"max_points_per_voxel must be > 0, got {max_points_per_voxel}"

        self.voxelizer = Voxelization(voxel_size,
                                      point_cloud_range,
                                      max_points_per_voxel,
                                      deterministic=False)

    def forward(self, points: torch.Tensor):
        assert isinstance(
            points,
            torch.Tensor), f"points must be a torch.Tensor, got {type(points)}"
        not_nan_mask = ~torch.isnan(points).any(dim=2)
        return self.voxelizer(points[not_nan_mask])


class DynamicVoxelizer(nn.Module):

    def __init__(self, voxel_size, point_cloud_range):
        super().__init__()
        self.voxelizer = Voxelization(voxel_size,
                                      point_cloud_range,
                                      max_num_points=-1)

    def forward(
            self,
            points: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        assert isinstance(
            points,
            torch.Tensor), f"points must be a torch.Tensor, got {type(points)}"

        batch_results = []
        for batch_idx in range(points.shape[0]):
            batch_points = points[batch_idx]
            not_nan_mask = ~torch.isnan(batch_points).any(dim=1)
            batch_non_nan_points = batch_points[not_nan_mask]
            batch_voxel_coords = self.voxelizer(batch_non_nan_points)
            # If any of the coords are -1, then the point is not in the voxel grid and should be discarded
            batch_voxel_coords_mask = (batch_voxel_coords != -1).all(dim=1)

            valid_batch_voxel_coords = batch_voxel_coords[
                batch_voxel_coords_mask]
            valid_batch_non_nan_points = batch_non_nan_points[
                batch_voxel_coords_mask]

            batch_results.append((valid_batch_non_nan_points, valid_batch_voxel_coords))
        return batch_results