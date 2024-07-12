from pytorch_lightning.loggers.logger import Logger
from models.base_models import BaseOptimizationModel
from models.components.neural_reps import GigaChadOccFlowMLP, ModelOccFlowResult
from models.mini_batch_optimization.gigachad_nsf import ModelFlowResult
from .gigachad_nsf import (
    GigachadNSFModel,
    GigachadNSFOptimizationLoop,
    ChamferDistanceType,
    ChamferTargetType,
    ModelFlowResult,
    QueryDirection,
    GigaChadNSFPreprocessedInput,
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


@dataclass
class GigaChadOccFlowPreprocessedInput(GigaChadNSFPreprocessedInput):
    free_space_rays: list[FreeSpaceRays]


class GigachadOccFlowModel(GigachadNSFModel):

    def __init__(
        self,
        full_input_sequence: TorchFullFrameInputSequence,
        speed_threshold: float,
        chamfer_target_type: ChamferTargetType | str,
        chamfer_distance_type: ChamferDistanceType | str,
    ) -> None:
        super().__init__(
            full_input_sequence=full_input_sequence,
            speed_threshold=speed_threshold,
            chamfer_target_type=chamfer_target_type,
            chamfer_distance_type=chamfer_distance_type,
        )
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
        return AdditiveCosts(
            [
                self._k_step_cost(rep, 1, QueryDirection.FORWARD, speed_limit=self.speed_threshold),
                self._k_step_cost(rep, 1, QueryDirection.REVERSE, speed_limit=self.speed_threshold),
                self._k_step_cost(rep, 3, QueryDirection.FORWARD),
                self._k_step_cost(rep, 3, QueryDirection.REVERSE),
                self._free_space_regularization(rep),
                self._cycle_consistency(rep) * 0.01,
            ]
        )

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

    def log(self, logger: Logger, prefix: str, epoch_idx: int) -> None:
        min_x = torch.inf
        max_x = -torch.inf
        min_y = torch.inf
        max_y = -torch.inf
        for idx in range(len(self.full_input_sequence)):
            points = self.full_input_sequence.get_global_pc(idx)
            min_x = min(min_x, points[:, 0].min().item())
            max_x = max(max_x, points[:, 0].max().item())
            min_y = min(min_y, points[:, 1].min().item())
            max_y = max(max_y, points[:, 1].max().item())

        # List of points at 0.2m resolution
        x = np.arange(min_x, max_x, 0.2)
        y = np.arange(min_y, max_y, 0.2)
        x_idxes = np.arange(len(x))
        y_idxes = np.arange(len(y))

        xy_grid = np.array(np.meshgrid(x, y))
        xy_idx_grid = np.array(np.meshgrid(x_idxes, y_idxes))

        xys = xy_grid.T.reshape(-1, 2)
        xy_idxes = xy_idx_grid.T.reshape(-1, 2)

        def make_img(idx: int, z: float = 0.5):
            xyzs = np.concatenate([xys, np.full((xys.shape[0], 1), z)], axis=1)
            with torch.inference_mode():
                with torch.no_grad():
                    xyzs_torch = torch.tensor(xyzs, dtype=torch.float32, device="cuda")
                    occ_flow_res: ModelOccFlowResult = self.model(
                        xyzs_torch,
                        idx,
                        len(self.full_input_sequence),
                        QueryDirection.FORWARD,
                    )
            occupancy_bev_image = np.zeros((len(x), len(y)))
            occupancy_bev_image[xy_idxes[:, 0], xy_idxes[:, 1]] = occ_flow_res.occ.cpu().numpy()
            # Expand the image to 3 channels
            return torch.from_numpy(occupancy_bev_image.T.reshape(1, len(y), len(x)))

        idxes = [
            int(0.25 * len(self.full_input_sequence)),
            int(0.5 * len(self.full_input_sequence)),
            int(0.75 * len(self.full_input_sequence)),
        ]

        imgs = [make_img(idx) for idx in idxes]

        logger.experiment.add_image(f"{prefix}/occ/0.25", imgs[0], epoch_idx)
        logger.experiment.add_image(f"{prefix}/occ/0.50", imgs[1], epoch_idx)
        logger.experiment.add_image(f"{prefix}/occ/0.75", imgs[2], epoch_idx)


class GigachadOccFlowOptimizationLoop(GigachadNSFOptimizationLoop):

    def __init__(
        self,
        speed_threshold: float,
        chamfer_target_type: ChamferTargetType | str,
        chamfer_distance_type: ChamferDistanceType | str = ChamferDistanceType.BOTH_DIRECTION,
        model_class: type[BaseOptimizationModel] = GigachadOccFlowModel,
        *args,
        **kwargs,
    ):
        super().__init__(
            speed_threshold=speed_threshold,
            chamfer_target_type=chamfer_target_type,
            chamfer_distance_type=chamfer_distance_type,
            model_class=model_class,
            *args,
            **kwargs,
        )
