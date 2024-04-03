import torch
import time
from collections import defaultdict
import enum
import numpy as np


class EarlyStoppingMode(enum.Enum):
    MIN = "min"
    MAX = "max"


class EarlyStopping:
    def __init__(
        self,
        mode: EarlyStoppingMode = EarlyStoppingMode.MIN,
        min_delta: float = 0,
        patience: int = 10,
        percentage: bool = False,
    ):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = np.inf if mode == EarlyStoppingMode.MIN else -np.inf
        self.num_bad_epochs = 0
        self.percentage = percentage

    def _is_better(self, current_perf_metric: float) -> bool:
        if self.patience == 0:
            return True

        if not self.percentage:
            if self.mode == EarlyStoppingMode.MIN:
                return current_perf_metric < self.best - self.min_delta
            elif self.mode == EarlyStoppingMode.MAX:
                return current_perf_metric > self.best + self.min_delta
        else:
            if self.mode == EarlyStoppingMode.MIN:
                return current_perf_metric < self.best - (self.best * self.min_delta / 100)
            elif self.mode == EarlyStoppingMode.MAX:
                return current_perf_metric > self.best + (self.best * self.min_delta / 100)

    def step(self, perf_metric: float) -> bool:
        if self.patience == 0:
            return False

        if torch.isnan(perf_metric):
            return True

        if self._is_better(perf_metric):
            self.num_bad_epochs = 0
            self.best = perf_metric
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False
