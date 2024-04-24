import torch

from dataloaders import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence
from models.neural_reps import FastNSF, FastNSFPlusPlus
from .nsfp_model import NSFPModel
from pytorch_lightning.loggers import Logger
from typing import Optional


class FastNSFModel(NSFPModel):

    def forward_single(
        self, input_sequence: BucketedSceneFlowInputSequence, logger: Logger
    ) -> BucketedSceneFlowOutputSequence:
        with torch.inference_mode(False):
            with torch.enable_grad():
                return self.optimization_loop.optimize(
                    model=FastNSF(input_sequence).to(input_sequence.device).train(),
                    problem=input_sequence,
                    title="Optimizing FastNSF",
                    logger=logger,
                    leave=True,
                )


class FastNSFPlusPlusModel(NSFPModel):

    def __init__(
        self,
        iterations: int = 5000,
        patience: int = 100,
        min_delta: float = 0.00005,
        speed_threshold: float = 30.0 / 10.0,  # 30 m/s cap
        save_flow_every: Optional[int] = None,
    ) -> None:
        super().__init__(
            iterations=iterations,
            patience=patience,
            min_delta=min_delta,
            save_flow_every=save_flow_every,
        )
        self.speed_threshold = speed_threshold

    def forward_single(
        self, input_sequence: BucketedSceneFlowInputSequence, logger: Logger
    ) -> BucketedSceneFlowOutputSequence:
        with torch.inference_mode(False):
            with torch.enable_grad():
                return self.optimization_loop.optimize(
                    model=FastNSFPlusPlus(input_sequence, speed_threshold=self.speed_threshold)
                    .to(input_sequence.device)
                    .train(),
                    problem=input_sequence,
                    title="Optimizing FastNSF++",
                    logger=logger,
                    leave=True,
                )
