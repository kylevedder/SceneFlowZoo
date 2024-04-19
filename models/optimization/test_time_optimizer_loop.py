import torch
import tqdm
from ..neural_reps import BaseNeuralRep
from .cost_functions import BaseCostProblem
from .utils import EarlyStopping
from dataloaders import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence
from typing import Optional


class OptimizationLoop:

    def __init__(
        self,
        iterations: int = 5000,
        lr: float = 0.008,
        patience: int = 100,
        min_delta: float = 0.00005,
        weight_decay: float = 0,
        compile: bool = True,
    ):
        self.iterations = iterations
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.compile = compile

    def optimize(
        self,
        model: BaseNeuralRep,
        problem: BucketedSceneFlowInputSequence,
        patience: Optional[int] = None,
        min_delta: Optional[float] = None,
        title: Optional[str] = "Optimizing Neur Rep",
        leave: bool = False,
    ) -> BucketedSceneFlowOutputSequence:
        model = model.train()
        if self.compile:
            model = torch.compile(model)
        problem = problem.clone().detach().requires_grad_(True)

        if patience is None:
            patience = self.patience

        if min_delta is None:
            min_delta = self.min_delta

        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        lowest_cost = torch.inf
        best_output = None

        bar = range(self.iterations)
        if title is not None:
            bar = tqdm.tqdm(range(self.iterations), leave=leave, desc=title)
        for _ in bar:
            optimizer.zero_grad()
            cost_problem = model.optim_forward_single(problem)
            cost = cost_problem.cost()

            if cost.item() < lowest_cost:
                lowest_cost = cost.item()
                # Run in eval mode to avoid unnecessary computation
                with torch.inference_mode():
                    best_output = model.forward_single(problem)

            cost.backward()
            optimizer.step()

            if early_stopping.step(cost):
                break

            bar.set_postfix(cost=f"{cost.item():0.4f}")

        assert best_output is not None, "Best output is None; optimization failed"
        return best_output
