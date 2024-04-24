import torch

from dataloaders import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence
from models.neural_reps import GigaChadNSF
from .nsfp_model import NSFPModel
from pytorch_lightning.loggers import Logger


class GigaChadNSFModel(NSFPModel):

    def __init__(
        self,
        iterations: int = 5000,
        patience: int = 100,
        min_delta: float = 0.00005,
        lr: float = 0.008,
        save_flow_every: int | None = None,
        speed_threshold: float = 30.0 / 10.0,  # 30 m/s cap
    ) -> None:
        super().__init__(
            iterations=iterations,
            patience=patience,
            min_delta=min_delta,
            save_flow_every=save_flow_every,
            lr=lr,
        )
        self.speed_threshold = speed_threshold

    def _validate_input(self, batched_sequence: list[BucketedSceneFlowInputSequence]) -> None:
        for sequence in batched_sequence:
            assert len(sequence) >= 2, f"Expected sequence length of >= 2, but got {len(sequence)}."

    def forward_single(
        self, input_sequence: BucketedSceneFlowInputSequence, logger: Logger
    ) -> BucketedSceneFlowOutputSequence:
        with torch.inference_mode(False):
            with torch.enable_grad():
                return self.optimization_loop.optimize(
                    model=GigaChadNSF(input_sequence, speed_threshold=self.speed_threshold)
                    .to(input_sequence.device)
                    .train(),
                    problem=input_sequence,
                    title="Optimizing GigaChadNSFModel",
                    logger=logger,
                    leave=True,
                )
