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
import enum
from core_utils.model_saver import ModelStateDicts


class OptimizerType(enum.Enum):
    ADAM = "adam"
    ADAMW = "adamw"


class WholeBatchBatcher(AbstractBatcher):

    def __init__(self, full_sequence: TorchFullFrameInputSequence):
        self.full_sequence = full_sequence

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> TorchFullFrameInputSequence:
        return self.full_sequence

    def shuffle_minibatches(self, seed: int = 0):
        pass


class DummyProfiler:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self

    def step(self):
        pass

    def save_results(self, key: str):
        pass


class WholeBatchOptimizationLoop(BaseTorchModel):

    def __init__(
        self,
        model_class: type[BaseOptimizationModel],
        scheduler: SchedulerBuilder | dict[str, object] = SchedulerBuilder(
            "StoppingScheduler", {"early_stopping": EarlyStopping()}
        ),
        epochs: int = 5000,
        optimizer_type: OptimizerType | str = OptimizerType.ADAM,
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
        self.optimizer_type = OptimizerType(optimizer_type)
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

    def _load_from_checkpoint(
        self, model: BaseOptimizationModel, checkpoint: Path | None
    ) -> ModelStateDicts:
        if checkpoint is None:
            return ModelStateDicts.default()
        checkpoint_data = torch.load(checkpoint)
        assert "model" in checkpoint_data, "Checkpoint must contain a 'model' key"
        assert "optimizer" in checkpoint_data, "Checkpoint must contain an 'optimizer' key"
        assert "scheduler" in checkpoint_data, "Checkpoint must contain a 'scheduler' key"
        assert "epoch" in checkpoint_data, "Checkpoint must contain an 'epoch' key"

        model_state_dicts = ModelStateDicts(
            model=checkpoint_data["model"],
            optimizer=checkpoint_data["optimizer"],
            scheduler=checkpoint_data["scheduler"],
            epoch=checkpoint_data["epoch"],
        )

        model.load_state_dict(model_state_dicts.model)
        return model_state_dicts

    def inference_forward_single(
        self, input_sequence: TorchFullFrameInputSequence, logger: Logger
    ) -> TorchFullFrameOutputSequence:
        if self.eval_only:
            # Build the model without any gradient tracking
            assert (
                self.checkpoint is not None
            ), "Must provide checkpoint for evaluation; eval_only is True"
            model = self.model_class(**self._model_constructor_args(input_sequence))
            # Load from checkpoint if provided
            model_state_dicts = self._load_from_checkpoint(model, self.checkpoint)
            model = model.to(input_sequence.device).train()
            return self._forward_inference(model, input_sequence, logger)
        else:
            # Build the model with gradient tracking
            with torch.inference_mode(False):
                with torch.enable_grad():
                    model = self.model_class(**self._model_constructor_args(input_sequence))
                    # Load from checkpoint if provided
                    model_state_dicts = self._load_from_checkpoint(model, self.checkpoint)
                    title = f"Optimizing {self.model_class.__name__}" if self.verbose else None

                    model = model.to(input_sequence.device).train()
                    return self.optimize(
                        model=model,
                        full_batch=input_sequence,
                        model_state_dicts=model_state_dicts,
                        title=title,
                        logger=logger,
                        leave=True,
                    )

    def _setup_batcher(self, full_sequence: TorchFullFrameInputSequence) -> AbstractBatcher:
        return WholeBatchBatcher(full_sequence)

    def _setup_optimizer(
        self, model: BaseOptimizationModel, model_state_dicts: ModelStateDicts
    ) -> torch.optim.Optimizer:
        match self.optimizer_type:
            case OptimizerType.ADAM:
                optim = torch.optim.Adam(
                    model.parameters(), lr=self.lr, weight_decay=self.weight_decay
                )
            case OptimizerType.ADAMW:
                optim = torch.optim.AdamW(
                    model.parameters(), lr=self.lr, weight_decay=self.weight_decay
                )
            case _:
                raise ValueError(f"Invalid optimizer type: {self.optimizer_type}")

        if model_state_dicts.optimizer is not None:
            optim.load_state_dict(model_state_dicts.optimizer)
        return optim

    def _setup_profiler(self, profile: bool) -> torch.profiler.profile | DummyProfiler:
        if not profile:
            return DummyProfiler()

        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

        def save_results(key: str):
            results = profiler.key_averages().table(sort_by=key, row_limit=30)
            results_file = Path() / f"profile_results_{key}.txt"
            with open(results_file, "w") as f:
                f.write(results)
            print(f"Saved profiling results to {results_file}")

        profiler.save_results = save_results

        return profiler

    def _setup_epochs(self, model_state_dicts: ModelStateDicts) -> list[int]:
        return list(range(model_state_dicts.epoch, self.epochs))

    def _log_results(
        self,
        logger: Logger,
        model: BaseOptimizationModel,
        full_batch: TorchFullFrameInputSequence,
        scheduler: StoppingScheduler,
        epoch_idx: int,
        batch_cost: float,
    ) -> None:
        logger_prefix = f"{full_batch.sequence_log_id}/{full_batch.dataset_idx:06d}"
        logger.log_metrics(
            {
                f"{logger_prefix}/batch_cost": (batch_cost),
                f"{logger_prefix}/lr": np.mean(scheduler.get_last_lr()),
            },
            step=epoch_idx,
        )

        model.epoch_end_log(logger, logger_prefix, epoch_idx)

    def _forward_optimize(
        self,
        model: BaseOptimizationModel,
        batch: TorchFullFrameInputSequence,
        logger: Logger,
    ) -> BaseCostProblem:
        try:
            (output,) = model(ForwardMode.TRAIN, [batch], logger)
        except Exception as e:
            print(f"Error during training: {e}")
            raise e
        return output

    def _forward_inference(
        self,
        model: BaseOptimizationModel,
        full_batch: TorchFullFrameInputSequence,
        logger: Logger,
    ) -> TorchFullFrameOutputSequence:
        try:
            with torch.inference_mode():
                with torch.no_grad():
                    (output,) = model(
                        ForwardMode.VAL, [full_batch.detach().requires_grad_(False)], logger
                    )
        except Exception as e:
            print(f"Error during inference: {e}")
            raise e
        return output

    def _optimize_inner_loop(
        self,
        minibatch: TorchFullFrameInputSequence,
        model: BaseOptimizationModel,
        optimizer: torch.optim.Optimizer,
        logger: Logger,
    ) -> float:
        optimizer.zero_grad()
        cost_problem = self._forward_optimize(model, minibatch, logger)
        cost = cost_problem.cost()
        cost.backward()
        optimizer.step()
        return cost.item()

    def optimize(
        self,
        logger: Logger,
        model: BaseOptimizationModel,
        full_batch: TorchFullFrameInputSequence,
        model_state_dicts: ModelStateDicts = ModelStateDicts.default(),
        title: str | None = "Optimizing Neur Rep",
        leave: bool = False,
        profile: bool = False,
    ) -> TorchFullFrameOutputSequence:
        model = model.train()
        if self.compile_model:
            model = torch.compile(model)

        full_batch = full_batch.clone().detach().requires_grad_(True)

        optimizer = self._setup_optimizer(model, model_state_dicts)
        scheduler = self.scheduler_builder.to_scheduler(optimizer, model_state_dicts)
        batcher = self._setup_batcher(full_batch)

        lowest_cost = torch.inf
        best_output: TorchFullFrameOutputSequence = None

        # Profile the training step
        with self._setup_profiler(profile=profile) as prof:

            epoch_bar = tqdm.tqdm(
                self._setup_epochs(model_state_dicts),
                leave=leave,
                desc=title,
                disable=title is None,
                position=1,
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
                for minibatch_idx, minibatch in enumerate(minibatch_bar):
                    try:
                        cost = self._optimize_inner_loop(minibatch, model, optimizer, logger)
                    except Exception as e:
                        print(f"Error during optimization: {e}")
                        # Print stack trace

                        raise e
                    total_cost += cost
                    minibatch_bar.set_postfix(cost=f"{cost:0.6f}")
                    prof.step()

                del minibatch  # Free up memory from the last minibatch

                if self.save_flow_every is not None:
                    if epoch_idx % self.save_flow_every == 0 or epoch_idx == self.epochs - 1:
                        self._save_model_state(
                            model, optimizer, scheduler, full_batch.dataset_idx, epoch_idx, logger
                        )

                if total_cost < lowest_cost:
                    lowest_cost = total_cost
                    best_output = self._forward_inference(model, full_batch, logger)

                batch_cost = total_cost / len(batcher)
                should_exit = scheduler.step(batch_cost)

                self._log_results(
                    logger,
                    model,
                    full_batch,
                    scheduler,
                    epoch_idx,
                    batch_cost,
                )
                if should_exit:
                    break

        # Save the profiling results to a text file
        prof.save_results("self_cuda_memory_usage")

        print(f"Exiting after {epoch_idx} epochs with lowest cost of {lowest_cost}")

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
