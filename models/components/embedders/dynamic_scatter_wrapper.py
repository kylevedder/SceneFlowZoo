from mmcv.ops import DynamicScatter
from typing import List
import torch


class DynamicScatterWrapper(DynamicScatter):

    def __init__(self, voxel_size: List, point_cloud_range: List, average_points: bool):
        super().__init__(voxel_size, point_cloud_range, average_points)

    def forward(self, points: torch.Tensor, coors: torch.Tensor):

        if torch.cuda.is_available():
            return super().forward(points, coors)
        else:
            # CPU version
            # points: (N, C) of floating point values, coors: (N, 3) of integer values
            # We are averaging points values together if they fall into the same voxel
            return self.forward_cpu(points, coors)

    def forward_cpu(self, points: torch.Tensor, coors: torch.Tensor):
        # Calculate voxel indices for each point
        voxel_indices = coors.long()

        # Find unique voxels and their indices
        unique_voxels, inverse_indices = torch.unique(voxel_indices, return_inverse=True, dim=0)

        # Aggregate points that fall into the same voxel
        aggregated_points = torch.zeros((unique_voxels.size(0), points.size(1)), dtype=points.dtype)

        for i in range(points.size(1)):  # Iterate through point features
            # Accumulate points features for each voxel
            aggregated_points[:, i] = torch.zeros_like(aggregated_points[:, i]).scatter_add_(
                0, inverse_indices, points[:, i]
            )

        if self.average_points:
            # Compute counts for each voxel to average
            counts = torch.zeros((unique_voxels.size(0),), dtype=points.dtype).scatter_add_(
                0, inverse_indices, torch.ones_like(inverse_indices, dtype=points.dtype)
            )
            # Avoid division by zero
            counts = torch.where(counts > 0, counts, torch.ones_like(counts))
            aggregated_points = aggregated_points / counts.unsqueeze(-1)

        return aggregated_points, unique_voxels
