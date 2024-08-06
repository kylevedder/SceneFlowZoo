from pytorch_lightning.loggers import Logger, TensorBoardLogger
from models.base_models import BaseOptimizationModel
from models.components.neural_reps import GigaChadOccFlowMLP, ModelOccFlowResult, ActivationFn
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

from dataloaders import (
    TorchFullFrameInputSequence,
    TorchFullFrameOutputSequence,
    TorchFullFrameOutputSequenceWithDistance,
    FreeSpaceRays,
)
from models.components.optimization.cost_functions import BaseCostProblem
from bucketed_scene_flow_eval.interfaces import LoaderType
from pointclouds import to_fixed_array_torch
import torch
import numpy as np
from dataclasses import dataclass
from visualization.flow_to_rgb import flow_to_rgb
import tqdm


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
        model: torch.nn.Module = GigaChadOccFlowMLP(),
        sampling_type: str = "fixed",
        max_unroll: int = 3,
    ) -> None:
        super().__init__(full_input_sequence, speed_threshold, pc_target_type, pc_loss_type, model)
        self.max_unroll = max_unroll
        self.sampling_type = sampling_type

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

        def _make_random_ray_distances(
            free_space_rays: FreeSpaceRays, min_dist: float, max_dist: float
        ) -> torch.Tensor:
            dim = len(free_space_rays)
            # Make random tensor with values uniformly distributed between min_dist and max_dist
            random_tensor = (
                torch.rand(dim, device=free_space_rays.rays.device) * (max_dist - min_dist)
                + min_dist
            )
            return random_tensor[:, None]

        match self.sampling_type:
            case "fixed":
                distance_schedule = [0.3, 0.6, 0.98]
            case "random":
                distance_schedule = [np.nan, np.nan, 0.98]
            case _:
                raise ValueError(f"Unknown sampling type {self.sampling_type}")

        def _sample_at_distance(range_scaler: float) -> BaseCostProblem:

            costs = []

            for idx in range(len(rep)):
                free_space_rays = rep.free_space_rays[idx]
                if np.isnan(range_scaler):
                    free_space_pc = free_space_rays.get_freespace_pc(
                        _make_random_ray_distances(free_space_rays, 0.0, 0.98)
                    )
                else:
                    free_space_pc = free_space_rays.get_freespace_pc(range_scaler)

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
                self._k_step_cost(rep, self.max_unroll, QueryDirection.FORWARD, self.pc_loss_type),
                self._k_step_cost(rep, self.max_unroll, QueryDirection.REVERSE, self.pc_loss_type),
                self._cycle_consistency(rep) * 0.01,
                self._free_space_regularization(rep),
            ]
        )
        # fmt: on

    def _interpolate_position(
        self,
        p_occupied_matrix: torch.Tensor,
        first_occupied_idx: torch.Tensor,
        is_occupied_threshold: float,
        sample_step_size: float,
    ):
        p_free_matrix = 1.0 - p_occupied_matrix
        is_free_threshold = 1.0 - is_occupied_threshold
        # last unoccupied index is the first occupied index - 1. Perform a max with 0 to handle the
        # case where the first occupied index is 0
        last_unoccupied_idx = torch.maximum(
            first_occupied_idx - 1, torch.zeros_like(first_occupied_idx)
        )

        first_occupied_distance = first_occupied_idx * sample_step_size
        last_unoccupied_distance = last_unoccupied_idx * sample_step_size

        first_occupied_free_space_probabilities = p_free_matrix[
            torch.arange(p_free_matrix.shape[0]), first_occupied_idx
        ]
        last_unoccupied_free_space_probabilities = p_free_matrix[
            torch.arange(p_free_matrix.shape[0]), last_unoccupied_idx
        ]

        # Linearly interpolate between the two distances based on the free space probabilities.
        interpolated_distance = (
            last_unoccupied_distance
            + (first_occupied_distance - last_unoccupied_distance)
            * last_unoccupied_free_space_probabilities
        )
        # Handle the case where the free space probability starts below the threshold
        interpolated_distance = torch.where(
            # The last unoccupied p(free space) is below the threshold, i.e. collision from the beginning
            last_unoccupied_free_space_probabilities < is_free_threshold,
            last_unoccupied_distance,
            interpolated_distance,
        )
        # Handle the case where the free space probability never drops below the threshold
        interpolated_distance = torch.where(
            # The last occupied p(free space) is above the threshold, i.e. no collision
            first_occupied_free_space_probabilities > is_free_threshold,
            first_occupied_distance,
            interpolated_distance,
        )

        is_colliding_ray = first_occupied_free_space_probabilities <= is_free_threshold

        return interpolated_distance, is_colliding_ray

    def _nerf_style_ray_marching(
        self,
        p_occupied_matrix: torch.Tensor,
        is_occupied_threshold: float,
        sample_step_size: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        is_free_threshold = 1.0 - is_occupied_threshold
        # Clamp the values to 0-1
        p_occupied_matrix = torch.clip(p_occupied_matrix, 0, 1)
        p_free_matrix = 1.0 - p_occupied_matrix
        # Do integration via cumulative product
        p_free_matrix_cumprod = torch.cumprod(p_free_matrix, axis=1)

        # Find the first index where the probability of being free is less than the threshold
        # We do this by taking the is free mask and summing along the columns to get the index
        # of the transition from free to occupied
        is_free_mask_matrix = p_free_matrix_cumprod > is_free_threshold
        is_free_mask_cumsum = torch.cumsum(is_free_mask_matrix, axis=1)
        # Max along the columns to get the first index where the transition occurs
        first_occupied_idx = torch.max(is_free_mask_cumsum, axis=1).indices

        return self._interpolate_position(
            p_occupied_matrix, first_occupied_idx, is_occupied_threshold, sample_step_size
        )

    def _simple_above_threshold(
        self,
        p_occupied_matrix: torch.Tensor,
        is_occupied_threshold: float,
        sample_step_size: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute the first index that is above threshold
        is_above_threshold_mask = p_occupied_matrix > is_occupied_threshold
        first_occupied_idx = torch.argmax(is_above_threshold_mask.float(), axis=1)
        return self._interpolate_position(
            p_occupied_matrix, first_occupied_idx, is_occupied_threshold, sample_step_size
        )

    def _compute_depth_reconstruction(
        self,
        rep: GigaChadNSFPreprocessedInput,
        query_idx: int,
        max_range_m: float = 50,
        num_ray_samples: int = 500,
        is_occupied_threshold: float = 0.99,
        direction: QueryDirection = QueryDirection.FORWARD,
        style: str = "value",
    ) -> tuple[torch.Tensor, torch.Tensor]:

        lidar_endpoints = rep.get_full_global_pc(query_idx)  # Nx3
        lidar_origin = rep.get_global_position(query_idx)  # 3
        # Subtract the origin from the endpoints to get the direction
        lidar_direction = lidar_endpoints - lidar_origin  # Nx3
        # Make them unit vectors
        lidar_unit_vectors = torch.nn.functional.normalize(lidar_direction, p=2, dim=1)

        sample_step_size = max_range_m / num_ray_samples

        # N x num_ray_samples matrix for recording the occupancy
        p_occupied_matrix = torch.zeros(
            (lidar_endpoints.shape[0], num_ray_samples), dtype=lidar_endpoints.dtype
        )

        for sample_idx in range(num_ray_samples):
            sample_ray_distance = sample_idx * sample_step_size
            sample_ray_endpoints = lidar_origin + (sample_ray_distance * lidar_unit_vectors)

            model_res: ModelOccFlowResult = self.model(
                sample_ray_endpoints,
                query_idx,
                len(rep),
                direction,
            )

            # Move to the CPU and numpy
            occ = model_res.occ.cpu()
            p_occupied_matrix[:, sample_idx] = occ

        if style == "nerf":
            return self._nerf_style_ray_marching(
                p_occupied_matrix, is_occupied_threshold, sample_step_size
            )
        elif style == "value":
            return self._simple_above_threshold(
                p_occupied_matrix, is_occupied_threshold, sample_step_size
            )
        else:
            raise ValueError(f"Unknown style {style}")

    def _forward_single_noncausal(
        self, input_sequence: TorchFullFrameInputSequence, logger: Logger
    ) -> TorchFullFrameOutputSequenceWithDistance:
        assert (
            input_sequence.loader_type == LoaderType.NON_CAUSAL
        ), f"Expected non-causal, but got {input_sequence.loader_type}"

        padded_n = input_sequence.get_pad_n()
        rep = self._preprocess(input_sequence)

        ego_flows = []
        valid_flow_masks = []
        depths = []
        is_colliding_ray_masks = []

        # For each index from 0 to len - 2, get the flow
        for idx in tqdm.tqdm(range(len(rep) - 1), desc="Inference frame"):
            ego_flow, full_pc_mask = self._compute_ego_flow(rep, idx)
            depth, is_colliding_ray_mask = self._compute_depth_reconstruction(rep, idx)
            padded_ego_flow = to_fixed_array_torch(ego_flow, padded_n)
            padded_full_pc_mask = to_fixed_array_torch(full_pc_mask, padded_n)
            padded_depth = to_fixed_array_torch(depth, padded_n)
            padded_is_colliding_ray_mask = to_fixed_array_torch(is_colliding_ray_mask, padded_n)
            ego_flows.append(padded_ego_flow)
            valid_flow_masks.append(padded_full_pc_mask)
            depths.append(padded_depth)
            is_colliding_ray_masks.append(padded_is_colliding_ray_mask)

        return TorchFullFrameOutputSequenceWithDistance(
            ego_flows=torch.stack(ego_flows),
            valid_flow_mask=torch.stack(valid_flow_masks),
            distances=torch.stack(depths),
            is_colliding_mask=torch.stack(is_colliding_ray_masks),
        )

    def inference_forward_single(
        self, input_sequence: TorchFullFrameInputSequence, logger: Logger
    ) -> TorchFullFrameOutputSequenceWithDistance:

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

    def _model_class(self) -> type[BaseOptimizationModel]:
        return GigachadOccFlowModel


class GigachadOccFlowGaussianOptimizationLoop(GigachadOccFlowOptimizationLoop):

    def _model_constructor_args(
        self, full_input_sequence: TorchFullFrameInputSequence
    ) -> dict[str, any]:
        return super()._model_constructor_args(full_input_sequence) | dict(
            model=GigaChadOccFlowMLP(act_fn=ActivationFn.GAUSSIAN)
        )


class GigachadOccFlowSincOptimizationLoop(GigachadOccFlowOptimizationLoop):

    def _model_constructor_args(
        self, full_input_sequence: TorchFullFrameInputSequence
    ) -> dict[str, any]:
        return super()._model_constructor_args(full_input_sequence) | dict(
            model=GigaChadOccFlowMLP(act_fn=ActivationFn.SINC)
        )


class GigachadOccFlowSincDepth10OptimizationLoop(GigachadOccFlowSincOptimizationLoop):

    def _model_constructor_args(
        self, full_input_sequence: TorchFullFrameInputSequence
    ) -> dict[str, any]:
        return super()._model_constructor_args(full_input_sequence) | dict(
            model=GigaChadOccFlowMLP(num_layers=10), max_unroll=2
        )


class GigachadOccFlowSincDepth12OptimizationLoop(GigachadOccFlowSincOptimizationLoop):

    def _model_constructor_args(
        self, full_input_sequence: TorchFullFrameInputSequence
    ) -> dict[str, any]:
        return super()._model_constructor_args(full_input_sequence) | dict(
            model=GigaChadOccFlowMLP(num_layers=12), max_unroll=2
        )


class GigachadOccFlowSincDepth14OptimizationLoop(GigachadOccFlowSincOptimizationLoop):

    def _model_constructor_args(
        self, full_input_sequence: TorchFullFrameInputSequence
    ) -> dict[str, any]:
        return super()._model_constructor_args(full_input_sequence) | dict(
            model=GigaChadOccFlowMLP(num_layers=14), max_unroll=2
        )


class GigachadOccFlowSincDepth10RandomSampleOptimizationLoop(GigachadOccFlowSincOptimizationLoop):

    def _model_constructor_args(
        self, full_input_sequence: TorchFullFrameInputSequence
    ) -> dict[str, any]:
        return super()._model_constructor_args(full_input_sequence) | dict(
            model=GigaChadOccFlowMLP(num_layers=10), max_unroll=2, sampling_type="random"
        )
