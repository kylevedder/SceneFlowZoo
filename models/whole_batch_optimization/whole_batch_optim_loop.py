import torch
import torch.profiler
import tqdm
from models.components.optimization.cost_functions import BaseCostProblem
from models.components.optimization.utils import EarlyStopping
from models.components.optimization.schedulers import (
    StoppingScheduler,
    ReduceLROnPlateauWithFloorRestart,
    SchedulerBuilder,
)
from dataloaders import TorchFullFrameInputSequence, TorchFullFrameOutputSequence
from typing import Optional
import numpy as np
from pytorch_lightning.loggers import Logger
from pathlib import Path
from bucketed_scene_flow_eval.utils import save_pickle
from models import BaseTorchModel, BaseOptimizationModel, ForwardMode
from models import AbstractBatcher


class WholeBatchBatcher(AbstractBatcher):

    def __init__(self, full_sequence: TorchFullFrameInputSequence):
        self.full_sequence = full_sequence

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> TorchFullFrameInputSequence:
        return self.full_sequence

    def shuffle_minibatches(self, seed: int = 0):
        pass


class WholeBatchOptimizationLoop(BaseTorchModel):

    def __init__(
        self,
        model_class: type[BaseOptimizationModel],
        scheduler: SchedulerBuilder | dict[str, object] = SchedulerBuilder(
            "StoppingScheduler", {"early_stopping": EarlyStopping()}
        ),
        epochs: int = 5000,
        lr: float = 0.008,
        weight_decay: float = 0,
        compile_model: bool = False,
        save_flow_every: int | None = None,
        verbose: bool = True,
        checkpoint: Path | None = None,
        eval_only: bool = False,
    ):
        super().__init__()
        # Ensure model_class is a type that's a subtype of BaseWholeBatchOptimizationModel
        assert issubclass(
            model_class, BaseOptimizationModel
        ), f"model_class must be a subclass of BaseWholeBatchOptimizationModel, but got {model_class}"
        self.model_class = model_class

        if isinstance(scheduler, dict):
            scheduler = SchedulerBuilder(scheduler["name"], scheduler["args"])
        self.scheduler_builder = scheduler

        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay

        self.compile_model = compile_model
        self.save_flow_every = save_flow_every
        self.verbose = verbose
        self.checkpoint = checkpoint
        self.eval_only = eval_only

    def _save_model_state(
        self,
        model: BaseOptimizationModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        dataset_idx: int,
        epoch: int,
        logger: Logger,
    ) -> None:
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        }
        model_save_path = (
            Path(logger.log_dir)
            / f"dataset_idx_{dataset_idx:010d}"
            / f"epoch_{epoch:08d}_checkpoint.pth"
        )
        # Make parent directory if it doesn't exist
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, model_save_path)

    def _save_intermediary_results(
        self,
        model: BaseOptimizationModel,
        problem: TorchFullFrameInputSequence,
        logger: Logger,
        optimization_step: int,
    ) -> None:
        model_save_path = (
            Path(logger.log_dir)
            / f"dataset_idx_{problem.dataset_idx:010d}"
            / f"opt_step_{optimization_step:08d}_weights.pth"
        )
        torch.save(model.state_dict(), model_save_path)

        with torch.inference_mode():
            with torch.no_grad():
                (output,) = model(ForwardMode.VAL, [problem], logger)
        output: TorchFullFrameOutputSequence
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
        self, full_input_sequence: TorchFullFrameInputSequence
    ) -> dict[str, any]:
        return dict(full_input_sequence=full_input_sequence)

    def inference_forward_single(
        self, input_sequence: TorchFullFrameInputSequence, logger: Logger
    ) -> TorchFullFrameOutputSequence:
        # print(
        #     f"Inference forward single for {input_sequence.sequence_log_id} {input_sequence.dataset_idx} with input_sequence of type {type(input_sequence)}"
        # )

        with torch.inference_mode(False):
            with torch.enable_grad():
                model = self.model_class(**self._model_constructor_args(input_sequence))
                # Load checkpoint weights
                if self.checkpoint is not None:
                    print(f"Loading checkpoint from {self.checkpoint}")
                    model.load_state_dict(torch.load(self.checkpoint))
                model = model.to(input_sequence.device).train()

                title = f"Optimizing {self.model_class.__name__}" if self.verbose else None

                if self.eval_only:
                    assert (
                        self.checkpoint is not None
                    ), "Must provide checkpoint for evaluation; eval_only is True"
                    print("Running in eval mode")
                    with torch.inference_mode():
                        with torch.no_grad():
                            (output,) = model(
                                ForwardMode.VAL,
                                [input_sequence.detach().requires_grad_(False)],
                                logger,
                            )
                    return output

                return self.optimize(
                    model=model,
                    full_batch=input_sequence,
                    title=title,
                    logger=logger,
                    leave=True,
                )

    def _setup_batcher(self, full_sequence: TorchFullFrameInputSequence) -> AbstractBatcher:
        return WholeBatchBatcher(full_sequence)

    def _setup_optimizer(self, model: BaseOptimizationModel) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def optimize(
        self,
        logger: Logger,
        model: BaseOptimizationModel,
        full_batch: TorchFullFrameInputSequence,
        title: str | None = "Optimizing Neur Rep",
        leave: bool = False,
    ) -> TorchFullFrameOutputSequence:
        model = model.train()
        if self.compile_model:
            model = torch.compile(model)
        full_batch = full_batch.clone().detach().requires_grad_(True)

        optimizer = self._setup_optimizer(model)
        scheduler = self.scheduler_builder.to_scheduler(optimizer)
        batcher = self._setup_batcher(full_batch)

        lowest_cost = torch.inf
        best_output: TorchFullFrameOutputSequence = None

        # Profile the training step
        # with torch.profiler.profile(
        #     activities=[
        #         torch.profiler.ProfilerActivity.CUDA,
        #     ],
        #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=self.epochs * len(batcher)),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True,
        # ) as prof:

        epoch_bar = tqdm.tqdm(
            range(self.epochs), leave=leave, desc=title, disable=title is None, position=1
        )
        for epoch_idx in epoch_bar:
            batcher.shuffle_minibatches(seed=epoch_idx)
            total_cost = 0
            minibatch_bar = tqdm.tqdm(
                batcher,
                leave=False,
                position=2,
                desc="Minibatches",
                disable=(len(batcher) <= 1) or (title is None),
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
                # prof.step()

            if self.save_flow_every is not None:
                if epoch_idx % self.save_flow_every == 0 or epoch_idx == self.epochs - 1:
                    self._save_model_state(
                        model, optimizer, scheduler, full_batch.dataset_idx, epoch_idx, logger
                    )

            if total_cost < lowest_cost:
                lowest_cost = total_cost
                # Run in eval mode to avoid unnecessary computation
                with torch.inference_mode():
                    with torch.no_grad():
                        (best_output,) = model(ForwardMode.VAL, [full_batch.detach()], logger)

            batch_cost = total_cost / len(batcher)
            should_exit = scheduler.step(batch_cost)

            logger_prefix = f"{full_batch.sequence_log_id}/{full_batch.dataset_idx:06d}"
            logger.log_metrics(
                {
                    f"{logger_prefix}/batch_cost": (batch_cost),
                    f"{logger_prefix}/lr": np.mean(scheduler.get_last_lr()),
                },
                step=epoch_idx,
            )

            model.log(logger, logger_prefix, epoch_idx)

            epoch_bar.set_postfix(cost=f"{batch_cost:0.6f}")

            if should_exit:
                break

        # # Save the profiling results to a text file
        # with open("profile_results.txt", "w") as f:
        #     f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

        assert best_output is not None, "Best output is None; optimization failed"
        return best_output

    def loss_fn(
        self,
        input_batch: list[TorchFullFrameInputSequence],
        model_res: list[TorchFullFrameOutputSequence],
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "Whole batch test time optimizer is not meant to be used for training. Run under test."
        )
