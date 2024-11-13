import torch
from pytorch_lightning.loggers.logger import Logger
from dataloaders import TorchFullFrameOutputSequence
from dataloaders.dataclasses import TorchFullFrameInputSequence
from models import BaseTorchModel, BaseOptimizationModel, ForwardMode
from models.components.optimization.cost_functions import (
    BaseCostProblem,
    AdditiveCosts,
    TruncatedChamferLossProblem,
)
from models.components.neural_reps import NSFPRawMLP
from dataclasses import dataclass

from .whole_batch_optim_loop import WholeBatchOptimizationLoop


@dataclass
class WholeBatchNSFPPreprocessedInput:
    full_pc0: torch.Tensor
    full_pc0_mask: torch.Tensor
    masked_pc0: torch.Tensor

    pc0_ego_to_global: torch.Tensor

    full_pc1: torch.Tensor
    full_pc1_mask: torch.Tensor
    masked_pc1: torch.Tensor


class NSFPForwardOnlyModel(BaseOptimizationModel):

    def __init__(self, full_input_sequence: TorchFullFrameInputSequence) -> None:
        super().__init__(full_input_sequence)
        self.forward_model = NSFPRawMLP()

    def _preprocess(
        self, input_sequence: TorchFullFrameInputSequence
    ) -> WholeBatchNSFPPreprocessedInput:
        full_pc0, full_pc0_mask = input_sequence.get_full_global_pc(
            -2
        ), input_sequence.get_full_pc_mask(-2)

        pc0_ego_to_global = input_sequence.pc_poses_ego_to_global[-2]

        full_pc1, full_pc1_mask = input_sequence.get_full_global_pc(
            -1
        ), input_sequence.get_full_pc_mask(-1)

        return WholeBatchNSFPPreprocessedInput(
            full_pc0=full_pc0,
            full_pc0_mask=full_pc0_mask,
            masked_pc0=torch.unsqueeze(full_pc0[full_pc0_mask], 0)
            .clone()
            .detach()
            .requires_grad_(True),
            pc0_ego_to_global=pc0_ego_to_global,
            full_pc1=full_pc1,
            full_pc1_mask=full_pc1_mask,
            masked_pc1=torch.unsqueeze(full_pc1[full_pc1_mask], 0)
            .clone()
            .detach()
            .requires_grad_(True),
        )

    def optim_forward_single(
        self, input_sequence: TorchFullFrameInputSequence, logger: Logger
    ) -> BaseCostProblem:

        rep = self._preprocess(input_sequence)

        # Ensure rep.masked_pc0 has grad
        assert rep.masked_pc0.requires_grad, "rep.masked_pc0 must have requires_grad=True"

        pc0_flow: torch.Tensor = self.forward_model(rep.masked_pc0)
        assert pc0_flow.requires_grad, f"pc0_flow must have requires_grad=True"
        warped_pc0_points = rep.masked_pc0 + pc0_flow

        # Ensure that the flows and warped points have gradients

        assert warped_pc0_points.requires_grad, "warped_pc0_points must have requires_grad=True"

        return TruncatedChamferLossProblem(
            warped_pc=warped_pc0_points,
            target_pc=rep.masked_pc1,
        )

    def inference_forward_single(
        self, input_sequence: TorchFullFrameInputSequence, logger: Logger
    ) -> TorchFullFrameOutputSequence:
        rep = self._preprocess(input_sequence)

        global_flow_pc0: torch.Tensor = self.forward_model(rep.masked_pc0).squeeze(0)

        full_global_flow_pc0 = torch.zeros_like(rep.full_pc0)
        full_global_flow_pc0[rep.full_pc0_mask] = global_flow_pc0

        ego_flow = self.global_to_ego_flow(
            rep.full_pc0, full_global_flow_pc0, rep.pc0_ego_to_global
        )

        return TorchFullFrameOutputSequence(
            ego_flows=torch.unsqueeze(ego_flow, 0),
            valid_flow_mask=torch.unsqueeze(rep.full_pc0_mask, 0),
        )


class NSFPCycleConsistencyModel(NSFPForwardOnlyModel):

    def __init__(self, full_input_sequence: TorchFullFrameInputSequence) -> None:
        super().__init__(full_input_sequence)
        self.reverse_model = NSFPRawMLP()

    def optim_forward_single(
        self, input_sequence: TorchFullFrameInputSequence, logger: Logger
    ) -> BaseCostProblem:

        rep = self._preprocess(input_sequence)

        # Ensure rep.masked_pc0 has grad
        assert rep.masked_pc0.requires_grad, "rep.masked_pc0 must have requires_grad=True"

        pc0_flow: torch.Tensor = self.forward_model(rep.masked_pc0)
        assert pc0_flow.requires_grad, f"pc0_flow must have requires_grad=True"
        warped_pc0_points = rep.masked_pc0 + pc0_flow

        cyclic_flow: torch.Tensor = self.reverse_model(warped_pc0_points)
        unwarped_pc0_points = warped_pc0_points + cyclic_flow

        # Ensure that the flows and warped points have gradients

        assert warped_pc0_points.requires_grad, "warped_pc0_points must have requires_grad=True"
        assert cyclic_flow.requires_grad, "cyclic_flow must have requires_grad=True"
        assert unwarped_pc0_points.requires_grad, "unwarped_pc0_points must have requires_grad=True"

        return AdditiveCosts(
            costs=[
                TruncatedChamferLossProblem(
                    warped_pc=warped_pc0_points,
                    target_pc=rep.masked_pc1,
                ),
                TruncatedChamferLossProblem(
                    warped_pc=unwarped_pc0_points,
                    target_pc=rep.masked_pc0,
                ),
            ]
        )


class NSFPForwardOnlyOptimizationLoop(WholeBatchOptimizationLoop):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(model_class=NSFPForwardOnlyModel, *args, **kwargs)


class NSFPCycleConsistencyOptimizationLoop(WholeBatchOptimizationLoop):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(model_class=NSFPCycleConsistencyModel, *args, **kwargs)
