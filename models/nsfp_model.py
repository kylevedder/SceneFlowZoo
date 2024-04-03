import time

import torch
import torch.nn as nn

from dataloaders import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence
from .base_model import BaseModel
from models.optimization import OptimizationLoop
from models.neural_reps import NSFP


class NSFPModel(BaseModel):
    def __init__(
        self,
        sequence_length: int = 2,
        iterations: int = 5000,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.optimization_loop = OptimizationLoop(iterations=iterations)

    def _validate_input(self, batched_sequence: list[BucketedSceneFlowInputSequence]) -> None:
        assert (
            self.sequence_length == 2
        ), "This implementation only supports a sequence length of 2."

        for sequence in batched_sequence:
            assert (
                len(sequence) == self.sequence_length
            ), f"Expected sequence length of {self.sequence_length}, but got {len(sequence)}."

    def forward_single(
        self, input_sequence: BucketedSceneFlowInputSequence
    ) -> BucketedSceneFlowOutputSequence:
        with torch.inference_mode(False):
            with torch.enable_grad():
                return self.optimization_loop.optimize(
                    model=NSFP().to(input_sequence.device).train(), problem=input_sequence
                )
