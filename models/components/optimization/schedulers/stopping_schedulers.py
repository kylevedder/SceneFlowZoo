import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR
from models.components.optimization.utils import EarlyStopping


class StoppingScheduler(LRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        early_stopping: EarlyStopping | dict[str, object] = EarlyStopping(),
    ):
        super().__init__(optimizer)
        assert isinstance(
            optimizer, Optimizer
        ), f"optimizer must be an Optimizer object, but got {optimizer}"

        if isinstance(early_stopping, dict):
            early_stopping = EarlyStopping(**early_stopping)
        assert isinstance(
            early_stopping, EarlyStopping
        ), f"early_stopping must be an EarlyStopping object, but got {early_stopping}"

        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _initial_step(self):
        """
        Overriden from base class to avoid the explicit step() call.
        """
        self.optimizer._step_count = 0
        self._step_count = 0

    def step(self, metric: float) -> bool:  # type: ignore
        metric = float(metric)
        super().step()
        return self.early_stopping.step(metric)

    def get_lr(self):
        return [param_group["lr"] for param_group in self.optimizer.param_groups]


class ReduceLROnPlateauWithFloorRestart(StoppingScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        early_stopping: EarlyStopping | dict[str, object] = EarlyStopping(),
        reduction_factor: float = 0.2,
        reduction_patience: int = 200,
        reduction_eps: float = 1e-4,
        num_restarts: int = 1,
        restart_max_lr: float = 1e-5,
        restart_min_lr: float = 1e-7,
    ):
        super().__init__(optimizer, early_stopping)
        # Ensure that the optimizer has consistent lr across all param groups, and it's the given max lr
        assert all(
            param_group["lr"] == restart_max_lr for param_group in optimizer.param_groups
        ), "Optimizer must have max lr across all param groups to start with"

        self.reduction_factor = reduction_factor
        self.reduction_patience = reduction_patience
        self.reduction_eps = reduction_eps
        self.num_restarts = num_restarts

        self.restart_max_lr = restart_max_lr
        self.restart_min_lr = restart_min_lr

        self.restart_counter = 0
        self.best: float | None = None
        self.last_epoch = 0
        self.num_bad_epochs = 0
        self.track_early_stopping = False

    def _set_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _is_better(self, current: float, best: float | None) -> bool:
        if best is None:
            return True
        return current < best - self.reduction_eps

    def _update_lr(self):
        # If any of the lrs are below the min_lr and restart_counter is less than num_restarts them to the initial lr
        under_threshold = any(
            group["lr"] < self.restart_min_lr for group in self.optimizer.param_groups
        )

        if under_threshold:
            if self.restart_counter < self.num_restarts:
                # Perform restart
                self.restart_counter += 1
                self.num_bad_epochs = 0
                self._set_lr(self.restart_max_lr)
                return
            else:
                # We are not cranking down the LR anymore, and we don't have any restarts left,
                # so we can start tracking early stopping.
                self.track_early_stopping = True
                return

        # If we're not under the threshold, then we can reduce the lr
        new_lr = self.optimizer.param_groups[0]["lr"] * self.reduction_factor
        self._set_lr(new_lr)

    def step(self, metrics: float) -> bool:  # type: ignore
        current = float(metrics)

        self.last_epoch += 1

        if self._is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.reduction_patience:
            self._update_lr()
            self.num_bad_epochs = 0

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

        if self.track_early_stopping:
            return self.early_stopping.step(current)
        return False
