import torch

from dataloaders import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence
from models.neural_reps import FastNSF, Liu2024
from .nsfp_model import NSFPModel
from dataclasses import dataclass


class Liu2024Model(NSFPModel):

    def _validate_input(self, batched_sequence: list[BucketedSceneFlowInputSequence]) -> None:
        for sequence in batched_sequence:
            assert len(sequence) == 3, f"Expected sequence length of 3, but got {len(sequence)}."

    def forward_single(
        self, input_sequence: BucketedSceneFlowInputSequence
    ) -> BucketedSceneFlowOutputSequence:

        # Taken from first party code: https://github.com/shenqildr/Multi-Frame-Neural-Scene-Flow/blob/c35a835f1cf929af4c5dbd89d847532c61b6768d/optimization.py#L645
        min_delta = 0.0001

        # Taken from email correspondence with authors
        forward_early_patience = 10
        reverse_early_patience = 10
        fusion_early_patience = 30

        assert (
            len(input_sequence) == 3
        ), f"Expected sequence length of 3, but got {len(input_sequence)}."
        forward_input = input_sequence.slice(1, 3)
        reverse_input = input_sequence.slice(0, 2).reverse()

        assert (
            len(forward_input) == 2
        ), f"Expected sequence length of 2, but got {len(forward_input)}."
        assert (
            len(reverse_input) == 2
        ), f"Expected sequence length of 2, but got {len(reverse_input)}."
        with torch.inference_mode(False):
            with torch.enable_grad():
                forward_res = self.optimization_loop.optimize(
                    model=FastNSF(forward_input).to(input_sequence.device).train(),
                    problem=forward_input,
                    title="Optimizing Forward Flow",
                    patience=forward_early_patience,
                    min_delta=min_delta,
                )
                reverse_res = self.optimization_loop.optimize(
                    model=FastNSF(reverse_input).to(input_sequence.device).train(),
                    problem=reverse_input,
                    title="Optimizing Reverse Flow",
                    patience=reverse_early_patience,
                    min_delta=min_delta,
                )
                return self.optimization_loop.optimize(
                    model=Liu2024(
                        input_sequence=forward_input,
                        forward_res=forward_res,
                        reverse_res=reverse_res,
                    )
                    .to(input_sequence.device)
                    .train(),
                    problem=forward_input,
                    title="Optimizing Fusion Flow",
                    patience=fusion_early_patience,
                    min_delta=min_delta,
                )
