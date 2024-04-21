import torch

from dataloaders import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence
from models.neural_reps import FastNSF, FastNSFPlusPlus
from .nsfp_model import NSFPModel


class FastNSFModel(NSFPModel):

    def forward_single(
        self, input_sequence: BucketedSceneFlowInputSequence
    ) -> BucketedSceneFlowOutputSequence:
        with torch.inference_mode(False):
            with torch.enable_grad():
                return self.optimization_loop.optimize(
                    model=FastNSF(input_sequence).to(input_sequence.device).train(),
                    problem=input_sequence,
                    title="Optimizing FastNSF",
                )


class FastNSFPlusPlusModel(NSFPModel):

    def __init__(
        self,
        iterations: int = 5000,
        patience: int = 100,
        min_delta: float = 0.00005,
        speed_threshold: float = 60.0 / 10.0,  # 60 m/s cap
    ) -> None:
        super().__init__(iterations=iterations, patience=patience, min_delta=min_delta)
        self.speed_threshold = speed_threshold

    def forward_single(
        self, input_sequence: BucketedSceneFlowInputSequence
    ) -> BucketedSceneFlowOutputSequence:
        with torch.inference_mode(False):
            with torch.enable_grad():
                return self.optimization_loop.optimize(
                    model=FastNSFPlusPlus(input_sequence, speed_threshold=self.speed_threshold)
                    .to(input_sequence.device)
                    .train(),
                    problem=input_sequence,
                    title="Optimizing FastNSF++",
                )
