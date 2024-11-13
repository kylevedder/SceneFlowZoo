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


class ChamferDistanceType(enum.Enum):
    BOTH_DIRECTION = "both_direction"
    FORWARD_ONLY = "forward_only"


@dataclass
class TruncatedChamferLossProblem(BaseCostProblem):
    warped_pc: torch.Tensor
    target_pc: torch.Tensor
    distance_threshold: Optional[float] = 2.0
    distance_type: ChamferDistanceType = ChamferDistanceType.FORWARD_ONLY

    def __post_init__(self):
        # Ensure that the PCs both have gradients enabled.
        assert self.warped_pc.requires_grad, "warped_pc must have requires_grad=True"
        assert self.target_pc.requires_grad, "target_pc must have requires_grad=True"

    def _get_x_y(self) -> tuple[torch.Tensor, torch.Tensor]:
        warped_pc = self.warped_pc
        target_pc = self.target_pc
        if warped_pc.ndim == 2:
            warped_pc = warped_pc.unsqueeze(0)
        if target_pc.ndim == 2:
            target_pc = target_pc.unsqueeze(0)

        x = warped_pc
        y = target_pc

        assert x.ndim == 3, f"x.ndim = {x.ndim}, not 3; shape = {x.shape}"
        assert y.ndim == 3, f"y.ndim = {y.ndim}, not 3; shape = {y.shape}"
        assert x.shape[2] == 3, f"x.shape[2] = {x.shape[2]}, not 3"
        assert y.shape[2] == 3, f"y.shape[2] = {y.shape[2]}, not 3"
        assert (
            x.shape[0] == y.shape[0] == 1
        ), f"x.shape[0] = {x.shape[0]}, y.shape[0] = {y.shape[0]}"
        return x, y

    def _make_lengths(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_lengths = torch.tensor([x.shape[1]], dtype=torch.long, device=x.device)
        y_lengths = torch.tensor([y.shape[1]], dtype=torch.long, device=y.device)
        return x_lengths, y_lengths

    def _single_pc_chamfer_knn_run(
        self, x: torch.Tensor, y: torch.Tensor, x_lengths: torch.Tensor, y_lengths: torch.Tensor
    ) -> torch.Tensor:
        x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
        cham_x = x_nn.dists[..., 0]
        if self.distance_threshold is not None:
            cham_x[cham_x >= self.distance_threshold] = 0.0
        return cham_x.mean()

    def _add_distances(self, cham_x: torch.Tensor, cham_y: torch.Tensor) -> torch.Tensor:
        return cham_x.mean() + cham_y.mean()

    def base_cost(self) -> torch.Tensor:
        x, y = self._get_x_y()
        x_lengths, y_lengths = self._make_lengths(x, y)
        x_mean = self._single_pc_chamfer_knn_run(x, y, x_lengths, y_lengths)

        if self.distance_type == ChamferDistanceType.FORWARD_ONLY:
            # Scaled by 2 to roughly match the magnitude of the bidirectional case
            return x_mean * 2
        elif self.distance_type == ChamferDistanceType.BOTH_DIRECTION:
            y_mean = self._single_pc_chamfer_knn_run(y, x, y_lengths, x_lengths)
            return x_mean + y_mean

    def __repr__(self) -> str:
        return f"TruncatedChamferLossProblem({self.base_cost()})"
