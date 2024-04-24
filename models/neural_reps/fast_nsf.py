import torch
import torch.nn as nn
from .base_neural_rep import BaseNeuralRep
from .nsfp_raw_mlp import NSFPRawMLP
from .nsfp import NSFPPreprocessedInput, NSFPForwardOnly

from dataloaders import BucketedSceneFlowInputSequence
from models.optimization.utils import EarlyStopping
from models.optimization.cost_functions import (
    BaseCostProblem,
    AdditiveCosts,
    DistanceTransform,
    DistanceTransformLossProblem,
    SpeedRegularizer,
)
from pytorch_lightning.loggers import Logger


class FastNSF(NSFPForwardOnly):

    def __init__(self, input_sequence: BucketedSceneFlowInputSequence):
        super().__init__()
        preprocess_result = self._preprocess(input_sequence.clone().detach())
        self.dt = DistanceTransform.from_pointclouds(
            preprocess_result.masked_pc0.clone().detach(),
            preprocess_result.masked_pc1.clone().detach(),
        )

        print("DT:", self.dt)

        self._prep_neural_prior(self.forward_model)

    def _prep_neural_prior(self, model: nn.Module):
        """
        Taken from
        https://github.com/Lilac-Lee/FastNSF/blob/386ab3862be22a09542570abc7032e46fcea0802/optimization.py#L393-L397
        """

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)

        model.apply(init_weights)
        for param in model.parameters():
            param.requires_grad = True

    def optim_forward_single(
        self,
        input_sequence: BucketedSceneFlowInputSequence,
        optim_step: int,
        early_stopping: EarlyStopping,
        logger: Logger,
    ) -> BaseCostProblem:
        rep = self._preprocess(input_sequence)

        # Ensure rep.masked_pc0 has grad
        assert rep.masked_pc0.requires_grad, "rep.masked_pc0 must have requires_grad=True"

        pc0_flow: torch.Tensor = self.forward_model(rep.masked_pc0)
        assert pc0_flow.requires_grad, f"pc0_flow must have requires_grad=True"
        warped_pc0_points = rep.masked_pc0 + pc0_flow

        # Ensure that the flows and warped points have gradients

        assert warped_pc0_points.requires_grad, "warped_pc0_points must have requires_grad=True"

        return DistanceTransformLossProblem(
            dt=self.dt,
            pc=warped_pc0_points,
        )


class FastNSFPlusPlus(FastNSF):

    def __init__(self, input_sequence: BucketedSceneFlowInputSequence, speed_threshold: float):
        super().__init__(input_sequence)
        self.speed_threshold = speed_threshold

    def optim_forward_single(
        self,
        input_sequence: BucketedSceneFlowInputSequence,
        optim_step: int,
        early_stopping: EarlyStopping,
        logger: Logger,
    ) -> BaseCostProblem:
        rep = self._preprocess(input_sequence)

        # Ensure rep.masked_pc0 has grad
        assert rep.masked_pc0.requires_grad, "rep.masked_pc0 must have requires_grad=True"

        pc0_flow: torch.Tensor = self.forward_model(rep.masked_pc0)
        assert pc0_flow.requires_grad, f"pc0_flow must have requires_grad=True"
        warped_pc0_points = rep.masked_pc0 + pc0_flow

        # Ensure that the flows and warped points have gradients

        assert warped_pc0_points.requires_grad, "warped_pc0_points must have requires_grad=True"

        return AdditiveCosts(
            [
                DistanceTransformLossProblem(
                    dt=self.dt,
                    pc=warped_pc0_points,
                ),
                SpeedRegularizer(
                    flow=pc0_flow.squeeze(0),
                    speed_threshold=self.speed_threshold,
                ),
            ]
        )
