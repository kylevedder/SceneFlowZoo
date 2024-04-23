from dataclasses import dataclass

from dataloaders import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence
from .nsfp_raw_mlp import NSFPRawMLP, ActivationFn
from .fast_nsf import FastNSF
from .nsfp import NSFPPreprocessedInput
from models.optimization.cost_functions import DistanceTransformLossProblem, BaseCostProblem
import torch
from pytorch_lightning.loggers import Logger


class Liu2024FusionRawMLP(NSFPRawMLP):

    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 3,
        latent_dim: int = 128,
        act_fn: ActivationFn = ActivationFn.RELU,
        layer_size: int = 3,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            act_fn=act_fn,
            num_layers=layer_size,
        )


@dataclass
class Liu2024PreprocessedInput(NSFPPreprocessedInput):
    fusion_input_features: torch.Tensor


class Liu2024(FastNSF):
    def __init__(
        self,
        input_sequence: BucketedSceneFlowInputSequence,
        forward_res: BucketedSceneFlowOutputSequence,
        reverse_res: BucketedSceneFlowOutputSequence,
    ):
        self.forward_res = forward_res.clone().detach()
        self.reverse_res = reverse_res.clone().detach()
        super().__init__(input_sequence)
        self.fusion_model = Liu2024FusionRawMLP()

    def _preprocess(
        self, input_sequence: BucketedSceneFlowInputSequence
    ) -> Liu2024PreprocessedInput:

        assert (
            len(self.forward_res) == 1
        ), f"Expected sequence length of 1, but got {len(self.forward_res)}."
        assert (
            len(self.reverse_res) == 1
        ), f"Expected sequence length of 1, but got {len(self.reverse_res)}."

        full_forward_flow = self.forward_res.get_full_ego_flow(0)
        full_forward_valid = self.forward_res.get_full_flow_mask(0)

        reverse = self.reverse_res.reverse()
        full_reverse_flow = reverse.get_full_ego_flow(0)
        full_reverse_valid = reverse.get_full_flow_mask(0)

        fusion_input_features = torch.cat(
            [
                full_forward_flow[full_forward_valid],
                full_reverse_flow[full_reverse_valid],
            ],
            dim=1,
        ).unsqueeze(0)

        assert (
            fusion_input_features.shape[0] == 1
        ), f"Expected batch size of 1, but got {fusion_input_features.shape[0]}."
        assert (
            fusion_input_features.shape[2] == 6
        ), f"Expected input features to have 6 channels, but got {fusion_input_features.shape[2]}."

        nsfp_preprocessed = super()._preprocess(input_sequence)
        return Liu2024PreprocessedInput(
            fusion_input_features=fusion_input_features.clone().detach().requires_grad_(True),
            **vars(nsfp_preprocessed),
        )

    def optim_forward_single(
        self, input_sequence: BucketedSceneFlowInputSequence, logger: Logger
    ) -> BaseCostProblem:
        rep = self._preprocess(input_sequence)

        assert (
            rep.fusion_input_features.requires_grad
        ), "rep.fusion_input_features must have requires_grad=True"

        pc0_flow: torch.Tensor = self.fusion_model(rep.fusion_input_features)
        assert pc0_flow.requires_grad, f"pc0_flow must have requires_grad=True"
        warped_pc0_points = rep.masked_pc0 + pc0_flow

        # Ensure that the flows and warped points have gradients

        assert warped_pc0_points.requires_grad, "warped_pc0_points must have requires_grad=True"

        return DistanceTransformLossProblem(
            dt=self.dt,
            pc=warped_pc0_points,
        )

    def forward_single(
        self, input_sequence: BucketedSceneFlowInputSequence, logger: Logger
    ) -> BucketedSceneFlowOutputSequence:

        rep = self._preprocess(input_sequence)

        global_flow_pc0 = self.fusion_model(rep.fusion_input_features).squeeze(0)

        full_global_flow_pc0 = torch.zeros_like(rep.full_pc0)
        full_global_flow_pc0[rep.full_pc0_mask] = global_flow_pc0

        ego_flow = self.global_to_ego_flow(
            rep.full_pc0, full_global_flow_pc0, rep.pc0_ego_to_global
        )

        return BucketedSceneFlowOutputSequence(
            ego_flows=torch.unsqueeze(ego_flow, 0),
            valid_flow_mask=torch.unsqueeze(rep.full_pc0_mask, 0),
        )
