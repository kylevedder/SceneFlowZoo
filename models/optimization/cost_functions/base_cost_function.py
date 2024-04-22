from abc import ABC, abstractmethod
import torch
from dataclasses import dataclass


@dataclass
class BaseCostProblem(ABC):

    @abstractmethod
    def cost(self) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class AdditiveCosts(BaseCostProblem):
    costs: list[BaseCostProblem]

    def cost(self) -> torch.Tensor:
        assert len(self.costs) > 0, "No costs to add"
        return sum([cost.cost() for cost in self.costs])

    def __repr__(self) -> str:
        return f"AdditiveCosts({self.costs})"
