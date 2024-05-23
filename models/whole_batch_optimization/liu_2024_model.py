import torch

from dataloaders import TorchFullFrameInputSequence, TorchFullFrameOutputSequence
from models.base_models import BaseOptimizationModel
from models.components.neural_reps import NSFPRawMLP, Liu2024FusionRawMLP
from pytorch_lightning.loggers import Logger
from models import ForwardMode
from .whole_batch_optim_loop import WholeBatchOptimizationLoop
from .nsfp_model import NSFPCycleConsistencyModel, WholeBatchNSFPPreprocessedInput
from .fast_nsf_model import FastNSFModel, FastNSFModelOptimizationLoop
from models.components.optimization.cost_functions import (
    BaseCostProblem,
    TruncatedChamferLossProblem,
    DistanceTransformLossProblem,
)
from dataclasses import dataclass


@dataclass
class Liu2024PreprocessedInput(WholeBatchNSFPPreprocessedInput):
    fusion_input_features: torch.Tensor


@dataclass
class Liu2024BucketedSceneFlowInputSequence(TorchFullFrameInputSequence):
    forward_result: TorchFullFrameOutputSequence
    reverse_result: TorchFullFrameOutputSequence


class Liu2024FusionModel(FastNSFModel):

    def __init__(self, full_input_sequence: Liu2024BucketedSceneFlowInputSequence) -> None:
        assert isinstance(full_input_sequence, Liu2024BucketedSceneFlowInputSequence), (
            f"Expected sequence to be of type Liu2024BucketedSceneFlowInputSequence, "
            f"but got {type(full_input_sequence)}."
        )
        super().__init__(
            full_input_sequence=full_input_sequence,
        )
        self.fusion_model = Liu2024FusionRawMLP()

    @staticmethod
    def input_sequence_to_forward_problem(
        full_input_sequence: TorchFullFrameInputSequence,
    ) -> TorchFullFrameInputSequence:
        return full_input_sequence.slice(1, 3)

    @staticmethod
    def input_sequence_to_reverse_problem(
        full_input_sequence: TorchFullFrameInputSequence,
    ) -> TorchFullFrameInputSequence:
        return full_input_sequence.slice(0, 2).reverse()

    def _validate_liu_input(self, sequence: Liu2024BucketedSceneFlowInputSequence) -> None:
        assert isinstance(sequence, Liu2024BucketedSceneFlowInputSequence), (
            f"Expected sequence to be of type Liu2024BucketedSceneFlowInputSequence, "
            f"but got {type(sequence)}."
        )
        assert len(sequence) == 3, f"Expected sequence length of 3, but got {len(sequence)}."

    def _preprocess_liu_input(
        self, input_sequence: Liu2024BucketedSceneFlowInputSequence
    ) -> Liu2024PreprocessedInput:
        self._validate_liu_input(input_sequence)

        input_sequence.forward_result.get_full_ego_flow(0)

        assert (
            len(input_sequence.forward_result) == 1
        ), f"Expected sequence length of 1, but got {len(input_sequence.forward_result)}."
        assert (
            len(input_sequence.reverse_result) == 1
        ), f"Expected sequence length of 1, but got {len(input_sequence.reverse_result)}."

        full_forward_flow = input_sequence.forward_result.get_full_ego_flow(0)
        full_forward_valid = input_sequence.forward_result.get_full_flow_mask(0)

        reverse = input_sequence.reverse_result.reverse()
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
        self, input_sequence: Liu2024BucketedSceneFlowInputSequence, logger: Logger
    ) -> BaseCostProblem:

        rep = self._preprocess_liu_input(input_sequence)

        pc0_flow: torch.Tensor = self.fusion_model(rep.fusion_input_features)
        assert pc0_flow.requires_grad, f"pc0_flow must have requires_grad=True"
        warped_pc0_points = rep.masked_pc0 + pc0_flow

        # Ensure that the flows and warped points have gradients
        assert warped_pc0_points.requires_grad, "warped_pc0_points must have requires_grad=True"
        return DistanceTransformLossProblem(dt=self.dt, pc=warped_pc0_points)

    def inference_forward_single(
        self, input_sequence: Liu2024BucketedSceneFlowInputSequence, logger: Logger
    ) -> TorchFullFrameOutputSequence:
        rep = self._preprocess_liu_input(input_sequence)

        global_flow_pc0: torch.Tensor = self.fusion_model(rep.fusion_input_features).squeeze(0)

        full_global_flow_pc0 = torch.zeros_like(rep.full_pc0)
        full_global_flow_pc0[rep.full_pc0_mask] = global_flow_pc0

        ego_flow = self.global_to_ego_flow(
            rep.full_pc0, full_global_flow_pc0, rep.pc0_ego_to_global
        )

        return TorchFullFrameOutputSequence(
            ego_flows=torch.unsqueeze(ego_flow, 0),
            valid_flow_mask=torch.unsqueeze(rep.full_pc0_mask, 0),
        )


class Liu2024OptimizationLoop(WholeBatchOptimizationLoop):
    """
    This implements Liu 2024, which runs FastNSF twice, once forward and once backward,
    and uses the resulting flow vectors as input to a small fusion model.

    This is implemented by first computing the forward and reverse flows using
    stand alone FastNSF optimization loops, and then feeding the result into the
    """

    def __init__(self, *args, **kwargs):

        # Taken from email correspondence with authors
        forward_early_patience = 10
        reverse_early_patience = 10
        fusion_early_patience = 30

        super().__init__(
            model_class=Liu2024FusionModel,
            patience=fusion_early_patience,
            *args,
            **kwargs,
        )

        self.forward_optmization_loop = FastNSFModelOptimizationLoop(
            patience=forward_early_patience,
            *args,
            **kwargs,
        )

        self.reverse_optmization_loop = FastNSFModelOptimizationLoop(
            patience=reverse_early_patience,
            *args,
            **kwargs,
        )

    def inference_forward_single(
        self, input_sequence: TorchFullFrameInputSequence, logger: Logger
    ) -> TorchFullFrameOutputSequence:

        forward_input = Liu2024FusionModel.input_sequence_to_forward_problem(input_sequence)
        reverse_input = Liu2024FusionModel.input_sequence_to_reverse_problem(input_sequence)

        forward_result = self.forward_optmization_loop.inference_forward_single(
            forward_input, logger
        )
        reverse_result = self.reverse_optmization_loop.inference_forward_single(
            reverse_input, logger
        )

        extended_input_sequence = Liu2024BucketedSceneFlowInputSequence(
            forward_result=forward_result,
            reverse_result=reverse_result,
            **vars(input_sequence),
        )

        assert isinstance(extended_input_sequence, Liu2024BucketedSceneFlowInputSequence), (
            f"Expected sequence to be of type Liu2024BucketedSceneFlowInputSequence, "
            f"but got {type(extended_input_sequence)}."
        )

        return super().inference_forward_single(extended_input_sequence, logger)
