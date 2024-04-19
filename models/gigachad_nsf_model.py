import torch

from dataloaders import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence
from models.neural_reps import GigaChadNSF
from .nsfp_model import NSFPModel


class GigaChadNSFModel(NSFPModel):

    def _validate_input(self, batched_sequence: list[BucketedSceneFlowInputSequence]) -> None:
        for sequence in batched_sequence:
            assert len(sequence) >= 2, f"Expected sequence length of >= 2, but got {len(sequence)}."

    def forward_single(
        self, input_sequence: BucketedSceneFlowInputSequence
    ) -> BucketedSceneFlowOutputSequence:
        with torch.inference_mode(False):
            with torch.enable_grad():
                return self.optimization_loop.optimize(
                    model=GigaChadNSF(input_sequence).to(input_sequence.device).train(),
                    problem=input_sequence,
                    title="Optimizing GigaChadNSFModel",
                )
