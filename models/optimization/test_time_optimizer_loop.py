import torch
import tqdm
from ..neural_reps import BaseNeuralRep
from .cost_functions import BaseCostProblem
from .utils import EarlyStopping
from dataloaders import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence


class OptimizationLoop:

    def __init__(
        self,
        iterations: int = 5000,
        lr: float = 0.008,
        weight_decay: float = 0,
        min_delta: float = 0.00005,
    ):
        self.iterations = iterations
        self.lr = lr
        self.weight_decay = weight_decay
        self.min_delta = min_delta

    def optimize(
        self,
        model: BaseNeuralRep,
        problem: BucketedSceneFlowInputSequence,
        title: str = "Optimizing Neur Rep",
        leave: bool = False,
    ) -> BucketedSceneFlowOutputSequence:
        model = model.train()
        problem = problem.clone().detach().requires_grad_(True)
        early_stopping = EarlyStopping(patience=100, min_delta=self.min_delta)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        lowest_cost = torch.inf
        best_output = None

        bar = tqdm.tqdm(range(self.iterations), leave=leave, desc=title)
        for epoch in bar:
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
