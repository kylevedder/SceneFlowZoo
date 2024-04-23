import torch
import tqdm
from ..neural_reps import BaseNeuralRep
from .cost_functions import BaseCostProblem
from .utils import EarlyStopping
from dataloaders import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence
from typing import Optional
import numpy as np
from pytorch_lightning.loggers import Logger
from pathlib import Path
from bucketed_scene_flow_eval.utils import save_pickle


class OptimizationLoop:

    def __init__(
        self,
        iterations: int = 5000,
        lr: float = 0.008,
        patience: int = 100,
        min_delta: float = 0.00005,
        weight_decay: float = 0,
        compile: bool = True,
        save_flow_every: Optional[int] = None,
    ):
        self.iterations = iterations
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.compile = compile
        self.save_flow_every = save_flow_every

    def _save_intermediary_results(
        self,
        model: BaseNeuralRep,
        problem: BucketedSceneFlowInputSequence,
        logger: Logger,
        optimization_step: int,
    ) -> None:

        output = model.forward_single(problem, logger)
        ego_flows = output.to_ego_lidar_flow_list()
        raw_ego_flows = [
            (
                ego_flow.full_flow,
                ego_flow.mask,
            )
            for ego_flow in ego_flows
        ]
        save_path = (
            Path(logger.log_dir)
            / f"dataset_idx_{problem.dataset_idx:010d}"
            / f"opt_step_{optimization_step:08d}.pkl"
        )
        save_pickle(save_path, raw_ego_flows, verbose=True)

    def optimize(
        self,
        logger: Logger,
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
        for step, _ in enumerate(bar):
            optimizer.zero_grad()
            cost_problem = model.optim_forward_single(problem, logger)
            cost = cost_problem.cost()
            logger.log_metrics(
                {f"log/{problem.sequence_log_id}/{problem.dataset_idx:06d}": cost.item()}, step=step
            )

            if self.save_flow_every is not None and step % self.save_flow_every == 0:
                self._save_intermediary_results(model, problem, logger, step)

            if cost.item() < lowest_cost:
                lowest_cost = cost.item()
                # Run in eval mode to avoid unnecessary computation
                with torch.inference_mode():
                    best_output = model.forward_single(problem, logger)

            cost.backward()
            optimizer.step()

            if early_stopping.step(cost):
                break

            bar.set_postfix(cost=f"{cost.item():0.4f}")

        assert best_output is not None, "Best output is None; optimization failed"
        return best_output
