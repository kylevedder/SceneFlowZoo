import torch
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss import chamfer_distance
from typing import Optional
from .base_cost_function import BaseCostProblem
from dataclasses import dataclass
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds
from typing import Union
import enum
from torch_kdtree.nn_distance import TorchKDTree


@dataclass
class TruncatedKDTreeLossProblem(BaseCostProblem):
    warped_pc: torch.Tensor
    torch_kdtree: TorchKDTree
    distance_threshold: Optional[float] = 2.0

    def __post_init__(self):
        # Ensure that the PCs both have gradients enabled.
        assert self.warped_pc.requires_grad, "warped_pc must have requires_grad=True"

    def _get_query_pc(self) -> torch.Tensor:
        warped_pc = self.warped_pc
        assert warped_pc.ndim == 2, f"warped_pc.ndim = {warped_pc.ndim}, not 3"
        return warped_pc

    def _kd_tree_dists(self, query_points: torch.Tensor) -> torch.Tensor:
        cham_x, _ = self.torch_kdtree.query(query_points, nr_nns_searches=1)
        if self.distance_threshold is not None:
            cham_x[cham_x >= self.distance_threshold] = 0.0
        return cham_x.mean()

    def base_cost(self) -> torch.Tensor:
        query_pc = self._get_query_pc()
        kd_mean = self._kd_tree_dists(query_pc)

        return kd_mean * 2

    def __repr__(self) -> str:
        return f"TruncatedKDTreeLossProblem({self.base_cost()})"
