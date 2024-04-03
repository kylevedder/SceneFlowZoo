import torch
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss import chamfer_distance
from typing import Optional
from .base_cost_function import BaseCostProblem
from dataclasses import dataclass
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds
from typing import Union


def my_chamfer_fn(x: torch.Tensor, y: torch.Tensor):

    # Ensure that both inputs are of shape 1 x _ x 3
    assert x.ndim == 3, f"x.ndim = {x.ndim}, not 3; shape = {x.shape}"
    assert y.ndim == 3, f"y.ndim = {y.ndim}, not 3; shape = {y.shape}"
    assert x.shape[2] == 3, f"x.shape[2] = {x.shape[2]}, not 3"
    assert y.shape[2] == 3, f"y.shape[2] = {y.shape[2]}, not 3"
    assert x.shape[0] == y.shape[0] == 1, f"x.shape[0] = {x.shape[0]}, y.shape[0] = {y.shape[0]}"

    x_lengths = torch.Tensor([x.shape[1]]).to(x.device).long()
    y_lengths = torch.Tensor([y.shape[1]]).to(y.device).long()

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    # NOTE: truncated Chamfer distance.
    dist_thd = 2
    cham_x[cham_x >= dist_thd] = 0.0
    cham_y[cham_y >= dist_thd] = 0.0

    return cham_x.mean() + cham_y.mean()


@dataclass
class TruncatedChamferLossProblem(BaseCostProblem):
    warped_pc: torch.Tensor
    target_pc: torch.Tensor
    distance_threshold: Optional[float] = 2.0

    def __post_init__(self):
        # Ensure that the PCs both have gradients enabled.
        assert self.warped_pc.requires_grad, "warped_pc must have requires_grad=True"
        assert self.target_pc.requires_grad, "target_pc must have requires_grad=True"

    def cost(self) -> torch.Tensor:
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

        x_lengths = torch.Tensor([x.shape[1]]).to(x.device).long()
        y_lengths = torch.Tensor([y.shape[1]]).to(y.device).long()

        x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
        y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

        cham_x = x_nn.dists[..., 0]
        cham_y = y_nn.dists[..., 0]

        # NOTE: truncated Chamfer distance.
        if self.distance_threshold is not None:
            cham_x[cham_x >= self.distance_threshold] = 0.0
            cham_y[cham_y >= self.distance_threshold] = 0.0

        return cham_x.mean() + cham_y.mean()
