import torch
import torch.nn as nn
from mmcv.ops import Voxelization


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
        return self.voxelizer(points)
