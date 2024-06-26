import torch
import tqdm
from models.components.optimization.cost_functions import BaseCostProblem
from models.components.optimization.utils import EarlyStopping
from dataloaders import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence
from typing import Optional
import numpy as np
from pytorch_lightning.loggers import Logger
from pathlib import Path
from bucketed_scene_flow_eval.utils import save_pickle
from models import BaseModel, BaseOptimizationModel, ForwardMode
from models import AbstractBatcher


class WholeBatchBatcher(AbstractBatcher):

    def __init__(self, full_sequence: BucketedSceneFlowInputSequence):
        self.full_sequence = full_sequence

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> BucketedSceneFlowInputSequence:
        return self.full_sequence

    def shuffle_minibatches(self, seed: int = 0):
        pass


class WholeBatchOptimizationLoop(BaseModel):

    def __init__(
        self,
        model_class: type[BaseOptimizationModel],
        epochs: int = 5000,
        lr: float = 0.008,
        burn_in_steps: int = 0,
        patience: int = 100,
        min_delta: float = 0.00005,
        weight_decay: float = 0,
        compile_model: bool = False,
        save_flow_every: int | None = None,
        verbose: bool = True,
    ):
        super().__init__()
        # Ensure model_class is a type that's a subtype of BaseWholeBatchOptimizationModel
        assert issubclass(
            model_class, BaseOptimizationModel
        ), f"model_class must be a subclass of BaseWholeBatchOptimizationModel, but got {model_class}"
        self.model_class = model_class
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.burn_in_steps = burn_in_steps
        self.patience = patience
        self.min_delta = min_delta
        self.compile_model = compile_model
        self.save_flow_every = save_flow_every
        self.verbose = verbose

    def _save_intermediary_results(
        self,
        model: BaseOptimizationModel,
        problem: BucketedSceneFlowInputSequence,
        logger: Logger,
        optimization_step: int,
    ) -> None:
        with torch.inference_mode():
            with torch.no_grad():
                (output,) = model(ForwardMode.VAL, [problem], logger)
        output: BucketedSceneFlowOutputSequence
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
        save_pickle(save_path, raw_ego_flows, verbose=self.verbose)

    def _model_constructor_args(
        self, full_input_sequence: BucketedSceneFlowInputSequence
    ) -> dict[str, any]:
        return dict(full_input_sequence=full_input_sequence)

    def inference_forward_single(
        self, input_sequence: BucketedSceneFlowInputSequence, logger: Logger
    ) -> BucketedSceneFlowOutputSequence:
        # print(
        #     f"Inference forward single for {input_sequence.sequence_log_id} {input_sequence.dataset_idx} with input_sequence of type {type(input_sequence)}"
        # )

        with torch.inference_mode(False):
            with torch.enable_grad():
                model = self.model_class(**self._model_constructor_args(input_sequence))
                model = model.to(input_sequence.device).train()

                title = f"Optimizing {self.model_class.__name__}" if self.verbose else None

                return self.optimize(
                    model=model,
                    full_batch=input_sequence,
                    title=title,
                    logger=logger,
                    leave=True,
                )

    def _setup_batcher(self, full_sequence: BucketedSceneFlowInputSequence) -> AbstractBatcher:
        return WholeBatchBatcher(full_sequence)

    def optimize(
        self,
        logger: Logger,
        model: BaseOptimizationModel,
        full_batch: BucketedSceneFlowInputSequence,
        burn_in_steps: int | None = None,
        patience: int | None = None,
        min_delta: float | None = None,
        title: str | None = "Optimizing Neur Rep",
        leave: bool = False,
    ) -> BucketedSceneFlowOutputSequence:
        model = model.train()
        if self.compile_model:
            model = torch.compile(model)
        full_batch = full_batch.clone().detach().requires_grad_(True)

        if patience is None:
            patience = self.patience

        if min_delta is None:
            min_delta = self.min_delta

        if burn_in_steps is None:
            burn_in_steps = self.burn_in_steps

        early_stopping = EarlyStopping(
            burn_in_steps=burn_in_steps, patience=patience, min_delta=min_delta
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        batcher = self._setup_batcher(full_batch)

        lowest_cost = torch.inf
        best_output: BucketedSceneFlowOutputSequence = None
        epoch_bar = tqdm.tqdm(
            range(self.epochs), leave=leave, desc=title, disable=title is None, position=1
        )
        for epoch_idx in epoch_bar:
            batcher.shuffle_minibatches(seed=epoch_idx)
            total_cost = 0
            minibatch_bar = tqdm.tqdm(
                batcher, leave=False, position=2, desc="Minibatches", disable=len(batcher) <= 1
            )
            for minibatch in minibatch_bar:
                optimizer.zero_grad()
                (cost_problem,) = model(ForwardMode.TRAIN, [minibatch], logger)
                cost_problem: BaseCostProblem
                cost = cost_problem.cost()

                cost.backward()
                optimizer.step()

                total_cost += cost.item()
                minibatch_bar.set_postfix(cost=f"{cost.item():0.6f}")

            if self.save_flow_every is not None and epoch_idx % self.save_flow_every == 0:
                self._save_intermediary_results(model, full_batch, logger, epoch_idx)

            logger.log_metrics(
                {f"log/{full_batch.sequence_log_id}/{full_batch.dataset_idx:06d}": total_cost},
                step=epoch_idx,
            )
            epoch_bar.set_postfix(cost=f"{total_cost / len(batcher):0.6f}")

            if total_cost < lowest_cost:
                lowest_cost = total_cost
                # Run in eval mode to avoid unnecessary computation
                with torch.inference_mode():
                    with torch.no_grad():
                        (best_output,) = model(ForwardMode.VAL, [full_batch.detach()], logger)

            if early_stopping.step(total_cost):
                break

        assert best_output is not None, "Best output is None; optimization failed"
        return best_output

    def loss_fn(
        self,
        input_batch: list[BucketedSceneFlowInputSequence],
        model_res: list[BucketedSceneFlowOutputSequence],
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "Whole batch test time optimizer is not meant to be used for training. Run under test."
        )
