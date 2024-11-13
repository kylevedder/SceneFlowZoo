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
from .truncated_chamfer_loss import TruncatedChamferLossProblem, ChamferDistanceType


class KDTreeWrapper:

    def __init__(self, ref_points: torch.Tensor) -> None:
        if self._is_gpu():
            self.kd_tree = self._build_gpu(ref_points)
        else:
            self.kd_tree = self._build_cpu(ref_points)

    def _is_gpu(self) -> bool:
        return torch.cuda.is_available()

    def _build_gpu(self, ref_points: torch.Tensor):
        from torch_kdtree import build_kd_tree

        return build_kd_tree(ref_points)

    def _build_cpu(self, ref_points: torch.Tensor):
        from scipy.spatial import KDTree

        return KDTree(ref_points.detach().cpu().numpy())

    def _query_cpu(self, query_points: torch.Tensor, nr_nns_searches):
        return self.kd_tree.query(query_points.detach().cpu().numpy(), k=nr_nns_searches)

    def _query_gpu(self, query_points: torch.Tensor, nr_nns_searches):
        return self.kd_tree.query(query_points, nr_nns_searches=nr_nns_searches)

    def query(self, query_points: torch.Tensor, nr_nns_searches: int = 1):
        if self._is_gpu():
            return self._query_gpu(query_points, nr_nns_searches)
        else:
            return self._query_cpu(query_points, nr_nns_searches)


@dataclass
class TruncatedForwardKDTreeLossProblem(BaseCostProblem):
    warped_pc: torch.Tensor
    kdtree: KDTreeWrapper
    distance_threshold: Optional[float] = 2.0

    def __post_init__(self):
        # Ensure that the PCs both have gradients enabled.
        assert self.warped_pc.requires_grad, "warped_pc must have requires_grad=True"

    def _get_query_pc(self) -> torch.Tensor:
        warped_pc = self.warped_pc
        assert warped_pc.ndim == 2, f"warped_pc.ndim = {warped_pc.ndim}, not 3"
        return warped_pc

    def _kd_tree_dists(self, query_points: torch.Tensor) -> torch.Tensor:
        cham_x, _ = self.kdtree.query(query_points, nr_nns_searches=1)
        if self.distance_threshold is not None:
            cham_x[cham_x >= self.distance_threshold] = 0.0
        return cham_x.mean()

    def base_cost(self) -> torch.Tensor:
        query_pc = self._get_query_pc()
        kd_mean = self._kd_tree_dists(query_pc)

        return kd_mean * 2

    def __repr__(self) -> str:
        return f"TruncatedForwardKDTreeLossProblem({self.base_cost()})"


@dataclass
class TruncatedForwardBackwardKDTreeLossProblem(BaseCostProblem):

    def __init__(
        self,
        warped_pc: torch.Tensor,
        target_pc: torch.Tensor,
        kdtree: KDTreeWrapper,
        distance_threshold: float | None = 2.0,
    ):
        self.forward_kd_tree = TruncatedForwardKDTreeLossProblem(
            warped_pc=warped_pc, kdtree=kdtree, distance_threshold=distance_threshold
        )
        self.reverse_chamfer = TruncatedChamferLossProblem(
            warped_pc=target_pc,
            target_pc=warped_pc,
            distance_threshold=distance_threshold,
            distance_type=ChamferDistanceType.FORWARD_ONLY,
        )

    def base_cost(self) -> torch.Tensor:
        total_cost = self.forward_kd_tree.base_cost() + self.reverse_chamfer.base_cost()
        return total_cost / 2

    def __repr__(self) -> str:
        return f"TruncatedForwardBackwardKDTreeLossProblem({self.base_cost()})"
