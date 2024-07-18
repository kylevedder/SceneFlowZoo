from pytorch_lightning.loggers import Logger, TensorBoardLogger
from models.base_models import BaseOptimizationModel
from models.components.neural_reps import GigaChadOccFlowMLP, ModelOccFlowResult
from models.mini_batch_optimization.gigachad_nsf import ModelFlowResult
from .gigachad_nsf import (
    GigachadNSFModel,
    GigachadNSFOptimizationLoop,
    ChamferDistanceType,
    PointCloudTargetType,
    ModelFlowResult,
    QueryDirection,
    GigaChadNSFPreprocessedInput,
    PointCloudLossType,
)

from models.components.optimization.cost_functions import (
    BaseCostProblem,
    PassthroughCostProblem,
    AdditiveCosts,
)

from dataloaders import TorchFullFrameInputSequence, TorchFullFrameOutputSequence, FreeSpaceRays
from models.components.optimization.cost_functions import BaseCostProblem
from bucketed_scene_flow_eval.interfaces import LoaderType
from pointclouds import to_fixed_array_torch
import torch
import numpy as np
from dataclasses import dataclass
from visualization.flow_to_rgb import flow_to_rgb


@dataclass
class GigaChadOccFlowPreprocessedInput(GigaChadNSFPreprocessedInput):
    free_space_rays: list[FreeSpaceRays]


class GigachadOccFlowModel(GigachadNSFModel):

    def __init__(
        self,
        full_input_sequence: TorchFullFrameInputSequence,
        speed_threshold: float,
        pc_target_type: PointCloudTargetType | str,
        pc_loss_type: PointCloudLossType | str,
    ) -> None:
        super().__init__(full_input_sequence, speed_threshold, pc_target_type, pc_loss_type)
        self.model = GigaChadOccFlowMLP()

    def _make_expected_zero_flow(self, model_res: ModelFlowResult) -> BaseCostProblem:
        assert isinstance(
            model_res, ModelFlowResult
        ), f"Expected ModelFlowResult, but got {type(model_res)}"

        cost = torch.abs(model_res.flow).mean()
        return PassthroughCostProblem(cost)

    def _make_occupancy_cost(
        self, model_res: ModelOccFlowResult, expected_value: float
    ) -> BaseCostProblem:
        assert isinstance(
            model_res, ModelOccFlowResult
        ), f"Expected ModelOccFlowResult, but got {type(model_res)}"

        cost = torch.abs(expected_value - model_res.occ).mean()
        return PassthroughCostProblem(cost)

    def _is_occupied_cost(self, model_res: ModelOccFlowResult) -> list[BaseCostProblem]:
        return [self._make_occupancy_cost(model_res, 1.0)]

    def _preprocess(
        self, input_sequence: TorchFullFrameInputSequence
    ) -> GigaChadOccFlowPreprocessedInput:
        super_res = super()._preprocess(input_sequence)

        free_space_rays: list[FreeSpaceRays] = []
        for idx in range(len(input_sequence)):
            free_space_rays.append(input_sequence.get_global_free_space_rays(idx))

        return GigaChadOccFlowPreprocessedInput(free_space_rays=free_space_rays, **vars(super_res))

    def _free_space_regularization(self, rep: GigaChadOccFlowPreprocessedInput) -> BaseCostProblem:
        assert isinstance(
            rep, GigaChadOccFlowPreprocessedInput
        ), f"Expected GigaChadOccFlowPreprocessedInput, but got {type(rep)}"
        distance_schedule = [0.3, 0.6, 0.98]

        def _sample_at_distance(distance: float) -> BaseCostProblem:

            costs = []

            for idx in range(len(rep)):
                free_space_pc = rep.free_space_rays[idx].get_freespace_pc(distance)
                sequence_idx = rep.sequence_idxes[idx]

                model_res: ModelOccFlowResult = self.model(
                    free_space_pc, sequence_idx, rep.sequence_total_length, QueryDirection.FORWARD
                )

                costs.append(self._make_occupancy_cost(model_res, 0.0))
                costs.append(self._make_expected_zero_flow(model_res))

            return AdditiveCosts(costs)

        return AdditiveCosts([_sample_at_distance(d) for d in distance_schedule])

    def optim_forward_single(
        self, input_sequence: TorchFullFrameInputSequence, logger: Logger
    ) -> BaseCostProblem:
        assert isinstance(
            input_sequence, TorchFullFrameInputSequence
        ), f"Expected BucketedSceneFlowInputSequence, but got {type(input_sequence)}"

        assert (
            input_sequence.loader_type == LoaderType.NON_CAUSAL
        ), f"Expected non-causal, but got {input_sequence.loader_type}"

        rep = self._preprocess(input_sequence)
        # fmt: off
        return AdditiveCosts(
            [
                self._k_step_cost(rep, 1, QueryDirection.FORWARD, self.pc_loss_type, speed_limit=self.speed_threshold),
                self._k_step_cost(rep, 1, QueryDirection.REVERSE, self.pc_loss_type, speed_limit=self.speed_threshold),
                self._k_step_cost(rep, 3, QueryDirection.FORWARD, self.pc_loss_type),
                self._k_step_cost(rep, 3, QueryDirection.REVERSE, self.pc_loss_type),
                self._cycle_consistency(rep) * 0.01,
                self._free_space_regularization(rep),
            ]
        )
        # fmt: on

    def _forward_single_noncausal(
        self, input_sequence: TorchFullFrameInputSequence, logger: Logger
    ) -> TorchFullFrameOutputSequence:
        assert (
            input_sequence.loader_type == LoaderType.NON_CAUSAL
        ), f"Expected non-causal, but got {input_sequence.loader_type}"

        padded_n = input_sequence.get_pad_n()
        rep = self._preprocess(input_sequence)

        ego_flows = []
        valid_flow_masks = []

        # For each index from 0 to len - 2, get the flow
        for idx in range(len(rep) - 1):
            ego_flow, full_pc_mask = self._compute_ego_flow(rep, idx)
            padded_ego_flow = to_fixed_array_torch(ego_flow, padded_n)
            padded_full_pc_mask = to_fixed_array_torch(full_pc_mask, padded_n)
            ego_flows.append(padded_ego_flow)
            valid_flow_masks.append(padded_full_pc_mask)

        return TorchFullFrameOutputSequence(
            ego_flows=torch.stack(ego_flows),
            valid_flow_mask=torch.stack(valid_flow_masks),
        )

    def inference_forward_single(
        self, input_sequence: TorchFullFrameInputSequence, logger: Logger
    ) -> TorchFullFrameOutputSequence:

        assert (
            input_sequence.loader_type == LoaderType.NON_CAUSAL
        ), f"Expected non-causal, but got {input_sequence.loader_type}"

        return self._forward_single_noncausal(input_sequence, logger)

    def _log_occ_bev(
        self,
        logger: Logger,
        prefix: str,
        percent_query: float,
        epoch_idx: int,
        model_res: ModelOccFlowResult,
        x: np.ndarray,
        y: np.ndarray,
        xy_idxes: np.ndarray,
    ):
        occupancy_bev_image = np.zeros((len(x), len(y)))
        occupancy_bev_image[xy_idxes[:, 0], xy_idxes[:, 1]] = model_res.occ.cpu().numpy()
        occupancy_bev_torch = torch.from_numpy(occupancy_bev_image.T.reshape(1, len(y), len(x)))

        logger.experiment.add_image(
            f"{prefix}/occ/{percent_query:0.2f}", occupancy_bev_torch, epoch_idx
        )

    def _process_grid_results(
        self,
        logger: Logger,
        prefix: str,
        percent_query: float,
        epoch_idx: int,
        result: ModelFlowResult,
        x: np.ndarray,
        y: np.ndarray,
        xy_idxes: np.ndarray,
    ):
        super()._process_grid_results(
            logger, prefix, percent_query, epoch_idx, result, x, y, xy_idxes
        )
        self._log_occ_bev(logger, prefix, percent_query, epoch_idx, result, x, y, xy_idxes)


class GigachadOccFlowOptimizationLoop(GigachadNSFOptimizationLoop):

    def __init__(
        self,
        speed_threshold: float,
        pc_target_type: PointCloudTargetType | str,
        pc_loss_type: (
            PointCloudLossType | str
        ) = PointCloudLossType.TRUNCATED_KD_TREE_FORWARD_BACKWARD,
        model_class: type[BaseOptimizationModel] = GigachadOccFlowModel,
        *args,
        **kwargs,
    ):
        super().__init__(
            speed_threshold, pc_target_type, pc_loss_type, model_class, *args, **kwargs
        )
