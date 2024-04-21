import torch
from .base_cost_function import BaseCostProblem
from dataclasses import dataclass


@dataclass
class SpeedRegularizer(BaseCostProblem):
    """
    Applies an L1 + L2 speed penalty to the flow field speed after crossing the given speed threshold.
    """

    flow: torch.Tensor
    speed_threshold: float

    def __post_init__(self):
        # Ensure that the PCs both have gradients enabled.
        assert self.flow.requires_grad, "flow must have requires_grad=True"
        assert (
            self.flow.ndim == 2
        ), f"flow.ndim = {self.flow.ndim}, not 2; shape = {self.flow.shape}"

    def cost(self) -> torch.Tensor:
        speed = torch.norm(self.flow, dim=1)
        l1_cost = torch.nn.functional.relu(speed - self.speed_threshold)
        l2_cost = l1_cost**2
        total_cost = (l1_cost + l2_cost).sum()
        return total_cost
