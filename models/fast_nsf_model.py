import torch

from dataloaders import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence
from models.neural_reps import FastNSF
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
                )
