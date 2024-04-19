import time

import torch
import torch.nn as nn

from dataloaders import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence
from .base_model import BaseModel
from models.optimization import OptimizationLoop
from models.neural_reps import NSFPCycleConsistency


class NSFPModel(BaseModel):
    def __init__(
        self,
        iterations: int = 5000,
        patience: int = 100,
        min_delta: float = 0.00005,
    ) -> None:
        super().__init__()
        self.optimization_loop = OptimizationLoop(
            iterations=iterations, min_delta=min_delta, patience=patience
        )

    def _validate_input(self, batched_sequence: list[BucketedSceneFlowInputSequence]) -> None:
        for sequence in batched_sequence:
            assert len(sequence) == 2, f"Expected sequence length of 2, but got {len(sequence)}."

    def forward_single(
        self, input_sequence: BucketedSceneFlowInputSequence
    ) -> BucketedSceneFlowOutputSequence:
        with torch.inference_mode(False):
            with torch.enable_grad():
                return self.optimization_loop.optimize(
                    model=NSFPCycleConsistency().to(input_sequence.device).train(),
                    problem=input_sequence,
                )
