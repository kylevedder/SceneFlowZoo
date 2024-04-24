from abc import ABC, abstractmethod
import torch
from dataclasses import dataclass


@dataclass(kw_only=True)
class BaseCostProblem(ABC):
    cost_scalar: float = 1.0

    @abstractmethod
    def base_cost(self) -> torch.Tensor:
        raise NotImplementedError

    def cost(self) -> torch.Tensor:
        return self.base_cost() * self.cost_scalar

    def __mul__(self, other: float) -> "BaseCostProblem":
        self.cost_scalar = other
        return self


@dataclass
class AdditiveCosts(BaseCostProblem):
    costs: list[BaseCostProblem]

    def base_cost(self) -> torch.Tensor:
        assert len(self.costs) > 0, "No costs to add"
        return sum([cost.base_cost() for cost in self.costs])

    def __repr__(self) -> str:
        return f"AdditiveCosts({self.costs} * {self.cost_scalar})"


@dataclass
class PointwiseLossProblem(BaseCostProblem):
    pred: torch.Tensor
    target: torch.Tensor

    def __post_init__(self):
        assert (
            self.pred.shape == self.target.shape
        ), f"Shapes do not match: {self.pred.shape} != {self.target.shape}"
        assert self.pred.ndim == 2, f"Expected 2D tensor, got {self.pred.ndim}D tensor"

    def base_cost(self) -> torch.Tensor:
        # Mean of the L2 norm across dim 1
        l2_error = torch.norm(self.pred - self.target, dim=1)
        sigmoided_error = torch.sigmoid(l2_error)
        return sigmoided_error.mean()

    def __repr__(self) -> str:
        return f"PointwiseLoss({self.base_cost()} * {self.cost_scalar})"
