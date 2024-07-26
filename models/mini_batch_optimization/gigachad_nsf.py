from pytorch_lightning.loggers.logger import Logger
from .mini_batch_optim_loop import MiniBatchOptimizationLoop, MinibatchedSceneFlowInputSequence
from models.components.neural_reps import (
    GigaChadFlowMLP,
    QueryDirection,
    ModelFlowResult,
    ActivationFn,
    FourierTemporalEmbedding,
)
from dataloaders import TorchFullFrameInputSequence, TorchFullFrameOutputSequence
from dataclasses import dataclass
from models import BaseOptimizationModel
from models.components.optimization.cost_functions import (
    BaseCostProblem,
    DistanceTransform,
    TruncatedForwardKDTreeLossProblem,
    TruncatedForwardBackwardKDTreeLossProblem,
    AdditiveCosts,
    PointwiseLossProblem,
    DistanceTransformLossProblem,
    TruncatedChamferLossProblem,
    ChamferDistanceType,
    SpeedRegularizer,
    KDTreeWrapper,
)
from bucketed_scene_flow_eval.interfaces import LoaderType
from pointclouds import to_fixed_array_torch
import enum
import torch
import torch.nn as nn
import tqdm
from visualization.flow_to_rgb import flow_to_rgb
import numpy as np


class PointCloudLossType(enum.Enum):
    TRUNCATED_CHAMFER_FORWARD = "truncated_chamfer"
    TRUNCATED_CHAMFER_FORWARD_BACKWARD = "truncated_chamfer_forward_backward"
    TRUNCATED_KD_TREE_FORWARD = "truncated_kd_tree_forward"
    TRUNCATED_KD_TREE_FORWARD_BACKWARD = "truncated_kd_tree_forward_backward"


class PointCloudTargetType(enum.Enum):
    LIDAR = "lidar"
    LIDAR_CAMERA = "lidar_camera"


@dataclass
class GigaChadNSFPreprocessedInput:
    full_global_pcs: list[torch.Tensor]
    full_global_pcs_mask: list[torch.Tensor]
    full_global_auxillary_pcs: list[torch.Tensor | None]
    ego_to_globals: list[torch.Tensor]
    sequence_idxes: list[int]
    sequence_total_length: int

    def __post_init__(self):
        # All lengths should be the same
        assert (
            len(self.full_global_pcs) == len(self.full_global_pcs_mask) == len(self.ego_to_globals)
        ), f"Expected lengths to be the same, but got {len(self.full_global_pcs)}, {len(self.full_global_pcs_mask)}, {len(self.ego_to_globals)}"

        for pc, mask in zip(self.full_global_pcs, self.full_global_pcs_mask):
            assert pc.shape[0] == mask.shape[0], f"Expected {pc.shape[0]}, but got {mask.shape[0]}"

    def __len__(self):
        return len(self.full_global_pcs)

    def get_full_global_pc(self, idx: int) -> torch.Tensor:
        return self.full_global_pcs[idx]

    def get_global_lidar_auxillary_pc(self, idx: int, with_grad: bool = True) -> torch.Tensor:
        lidar_pc = self.full_global_pcs[idx]
        lidar_mask = self.full_global_pcs_mask[idx]
        valid_pc = lidar_pc[lidar_mask]

        # If the camera pc is not available, return the lidar pc
        auxillary_pc = self.full_global_auxillary_pcs[idx]
        assert auxillary_pc is not None, "Expected auxillary_pc to be available"
        # concatenate the camera pc to the lidar pc
        valid_pc = torch.cat([valid_pc, auxillary_pc], dim=0)

        if with_grad:
            return valid_pc.clone().detach().requires_grad_(True)
        return valid_pc.clone().detach()

    def get_global_lidar_pc(self, idx: int, with_grad: bool = True) -> torch.Tensor:
        lidar_pc = self.full_global_pcs[idx]
        lidar_mask = self.full_global_pcs_mask[idx]
        valid_lidar_pc = lidar_pc[lidar_mask]

        if with_grad:
            return valid_lidar_pc.clone().detach().requires_grad_(True)
        return valid_lidar_pc.clone().detach()

    def get_full_pc_mask(self, idx: int) -> torch.Tensor:
        mask = self.full_global_pcs_mask[idx]
        # Conver to bool tensor
        return mask.bool()


class GigachadNSFModel(BaseOptimizationModel):

    def __init__(
        self,
        full_input_sequence: TorchFullFrameInputSequence,
        speed_threshold: float,
        pc_target_type: PointCloudTargetType | str,
        pc_loss_type: PointCloudLossType | str,
        model: torch.nn.Module = GigaChadFlowMLP(),
    ) -> None:
        super().__init__(full_input_sequence)
        self.model = model
        self.speed_threshold = speed_threshold
        self.pc_target_type = PointCloudTargetType(pc_target_type)
        self.pc_loss_type = PointCloudLossType(pc_loss_type)

        self._prep_neural_prior(self.model)

        self.kd_trees: list[KDTreeWrapper] | None = None

    def _prep_kdtrees(self) -> list[KDTreeWrapper]:
        full_rep = self._preprocess(self.full_input_sequence)
        kd_trees = []
        for idx in tqdm.tqdm(range(len(full_rep)), desc="Building KD Trees"):
            match self.pc_target_type:
                case PointCloudTargetType.LIDAR:
                    target_pc = full_rep.get_global_lidar_pc(idx, with_grad=False)
                case PointCloudTargetType.LIDAR_CAMERA:
                    target_pc = full_rep.get_global_lidar_auxillary_pc(idx, with_grad=False)
                case _:
                    raise ValueError(f"Unknown point cloud target type: {self.pc_target_type}")

            kd_trees.append(KDTreeWrapper(target_pc))
        return kd_trees

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

    def _preprocess(
        self, input_sequence: TorchFullFrameInputSequence
    ) -> GigaChadNSFPreprocessedInput:

        full_global_pcs: list[torch.Tensor] = []
        full_global_pcs_mask: list[torch.Tensor] = []
        full_global_camera_pcs: list[torch.Tensor | None] = []
        ego_to_globals: list[torch.Tensor] = []

        for idx in range(len(input_sequence)):
            full_pc, full_pc_mask = input_sequence.get_full_global_pc(
                idx
            ), input_sequence.get_full_pc_mask(idx)

            full_global_camera_pc = input_sequence.get_global_auxillary_pc(idx)

            ego_to_global = input_sequence.pc_poses_ego_to_global[idx]

            full_global_pcs.append(full_pc)
            full_global_pcs_mask.append(full_pc_mask)
            full_global_camera_pcs.append(full_global_camera_pc)
            ego_to_globals.append(ego_to_global)

        if isinstance(input_sequence, MinibatchedSceneFlowInputSequence):
            sequence_idxes = list(
                range(
                    input_sequence.minibatch_idx, input_sequence.minibatch_idx + len(input_sequence)
                )
            )
            sequence_total_length = len(input_sequence.full_sequence)
        else:
            sequence_idxes = list(range(len(input_sequence)))
            sequence_total_length = len(input_sequence)

        return GigaChadNSFPreprocessedInput(
            full_global_pcs=full_global_pcs,
            full_global_pcs_mask=full_global_pcs_mask,
            full_global_auxillary_pcs=full_global_camera_pcs,
            ego_to_globals=ego_to_globals,
            sequence_idxes=sequence_idxes,
            sequence_total_length=sequence_total_length,
        )

    def _is_occupied_cost(self, model_res: ModelFlowResult) -> list[BaseCostProblem]:
        return []

    def _cycle_consistency(self, rep: GigaChadNSFPreprocessedInput) -> BaseCostProblem:
        cost_problems: list[BaseCostProblem] = []
        for idx in range(len(rep) - 1):
            pc = rep.get_global_lidar_pc(idx)
            sequence_idx = rep.sequence_idxes[idx]
            sequence_total_length = rep.sequence_total_length
            model_res_forward: ModelFlowResult = self.model(
                pc,
                sequence_idx,
                sequence_total_length,
                QueryDirection.FORWARD,
            )
            forward_flowed_pc = pc + model_res_forward.flow

            model_res_reverse: ModelFlowResult = self.model(
                forward_flowed_pc,
                sequence_idx + 1,
                sequence_total_length,
                QueryDirection.REVERSE,
            )
            round_trip_flowed_pc = forward_flowed_pc + model_res_reverse.flow

            cost_problems.append(PointwiseLossProblem(pred=round_trip_flowed_pc, target=pc))
            cost_problems.extend(self._is_occupied_cost(model_res_forward))
            cost_problems.extend(self._is_occupied_cost(model_res_reverse))
        return AdditiveCosts(cost_problems)

    def _get_kd_tree(self, rep: GigaChadNSFPreprocessedInput, rep_idx: int) -> KDTreeWrapper:
        global_idx = rep.sequence_idxes[rep_idx]
        if self.kd_trees is None:
            self.kd_trees = self._prep_kdtrees()
        return self.kd_trees[global_idx]

    def _process_k_step_subk(
        self,
        rep: GigaChadNSFPreprocessedInput,
        anchor_pc: torch.Tensor,
        anchor_idx: int,
        subk: int,
        query_direction: QueryDirection,
        loss_type: PointCloudLossType,
        speed_limit: float | None,
    ) -> tuple[BaseCostProblem, torch.Tensor]:
        sequence_idx = rep.sequence_idxes[anchor_idx]
        model_res: ModelFlowResult = self.model(
            anchor_pc,
            sequence_idx,
            rep.sequence_total_length,
            query_direction,
        )
        anchor_pc = anchor_pc + model_res.flow

        index_offset = (subk + 1) * query_direction.value

        def _get_target_pc() -> torch.Tensor:
            match self.pc_target_type:
                case PointCloudTargetType.LIDAR:
                    target_pc = rep.get_global_lidar_pc(anchor_idx + index_offset)
                case PointCloudTargetType.LIDAR_CAMERA:
                    target_pc = rep.get_global_lidar_auxillary_pc(anchor_idx + index_offset)
            return target_pc

        match loss_type:
            case PointCloudLossType.TRUNCATED_CHAMFER_FORWARD:
                problem: BaseCostProblem = TruncatedChamferLossProblem(
                    warped_pc=anchor_pc,
                    target_pc=_get_target_pc(),
                    distance_type=ChamferDistanceType.FORWARD_ONLY,
                )
            case PointCloudLossType.TRUNCATED_CHAMFER_FORWARD_BACKWARD:
                problem = TruncatedChamferLossProblem(
                    warped_pc=anchor_pc,
                    target_pc=_get_target_pc(),
                    distance_type=ChamferDistanceType.BOTH_DIRECTION,
                )
            case PointCloudLossType.TRUNCATED_KD_TREE_FORWARD:
                problem = TruncatedForwardKDTreeLossProblem(
                    warped_pc=anchor_pc, kdtree=self._get_kd_tree(rep, anchor_idx + index_offset)
                )
            case PointCloudLossType.TRUNCATED_KD_TREE_FORWARD_BACKWARD:
                problem = TruncatedForwardBackwardKDTreeLossProblem(
                    warped_pc=anchor_pc,
                    target_pc=_get_target_pc(),
                    kdtree=self._get_kd_tree(rep, anchor_idx + index_offset),
                )
            case _:
                raise ValueError(f"Unknown loss type: {loss_type}")

        problem_list = [problem]

        if speed_limit is not None:
            problem_list.append(
                SpeedRegularizer(
                    flow=model_res.flow.squeeze(0),
                    speed_threshold=speed_limit,
                )
            )

        problem_list.extend(self._is_occupied_cost(model_res))

        return AdditiveCosts(problem_list), anchor_pc

    def _k_step_cost(
        self,
        rep: GigaChadNSFPreprocessedInput,
        k: int,
        query_direction: QueryDirection,
        loss_type: PointCloudLossType,
        speed_limit: float | None = None,
    ) -> BaseCostProblem:
        assert k >= 1, f"Expected k >= 1, but got {k}"
        assert (
            len(rep) > k
        ), f"Requested {k} steps, but this requires at least {k + 1} frames; got {len(rep)} frames"

        costs = []

        if query_direction == QueryDirection.FORWARD:
            anchor_idx_range = range(len(rep) - k)
        elif query_direction == QueryDirection.REVERSE:
            anchor_idx_range = range(k, len(rep))

        for anchor_idx in anchor_idx_range:
            anchor_pc = rep.get_global_lidar_pc(anchor_idx)

            for subk in range(k):
                problem, anchor_pc = self._process_k_step_subk(
                    rep, anchor_pc, anchor_idx, subk, query_direction, loss_type, speed_limit
                )
                costs.append(problem)
        return AdditiveCosts(costs)

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
            ]
        )
        # fmt: on

    def _compute_ego_flow(
        self,
        rep: GigaChadNSFPreprocessedInput,
        query_idx: int,
        direction: QueryDirection = QueryDirection.FORWARD,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model_res: ModelFlowResult = self.model(
            rep.get_global_lidar_pc(query_idx),
            query_idx,
            len(rep),
            direction,
        )
        global_flow_pc = model_res.flow.squeeze(0)

        full_global_pc = rep.get_full_global_pc(query_idx)
        full_global_flow_pc = torch.zeros_like(full_global_pc)
        full_pc_mask = rep.get_full_pc_mask(query_idx)
        full_global_flow_pc[full_pc_mask] = global_flow_pc

        ego_flow = self.global_to_ego_flow(
            full_global_pc, full_global_flow_pc, rep.ego_to_globals[query_idx]
        )

        return ego_flow, full_pc_mask

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

    def _build_grid_sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        x = np.arange(min_x, max_x, 0.2)
        y = np.arange(min_y, max_y, 0.2)
        x_idxes = np.arange(len(x))
        y_idxes = np.arange(len(y))

        xy_grid = np.array(np.meshgrid(x, y))
        xy_idx_grid = np.array(np.meshgrid(x_idxes, y_idxes))

        xys = xy_grid.T.reshape(-1, 2)
        xy_idxes = xy_idx_grid.T.reshape(-1, 2)

        return x, y, xys, xy_idxes

    def _log_flow_bev(
        self,
        logger: Logger,
        prefix: str,
        percent_query: float,
        epoch_idx: int,
        model_res: ModelFlowResult,
        x: np.ndarray,
        y: np.ndarray,
        xy_idxes: np.ndarray,
    ):

        flow = model_res.flow.cpu().numpy()
        flow_bev_image = np.zeros((len(x), len(y), 3), dtype=np.uint8)
        flow_rgbs = flow_to_rgb(flow, flow_max_radius=0.15, background="bright")

        flow_bev_image[xy_idxes[:, 0], xy_idxes[:, 1]] = flow_rgbs
        flow_bev_torch = torch.from_numpy(flow_bev_image.transpose((2, 1, 0)))

        logger.experiment.add_image(
            f"{prefix}/flow/{percent_query:0.2f}", flow_bev_torch, epoch_idx
        )

    def _query_grid_sample(
        self,
        xys: np.ndarray,
        idx: int,
        z: float = 0.5,  # Meters
    ) -> ModelFlowResult:
        xyzs = np.concatenate([xys, np.full((xys.shape[0], 1), z)], axis=1)
        with torch.inference_mode():
            with torch.no_grad():
                xyzs_torch = torch.tensor(xyzs, dtype=torch.float32, device="cuda")
                model_res: ModelFlowResult = self.model(
                    xyzs_torch,
                    idx,
                    len(self.full_input_sequence),
                    QueryDirection.FORWARD,
                )

        return model_res

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
        self._log_flow_bev(logger, prefix, percent_query, epoch_idx, result, x, y, xy_idxes)

    def epoch_end_log(self, logger: Logger, prefix: str, epoch_idx: int) -> None:
        if (epoch_idx % 5) != 0:
            # Skip logging grid samples for most epochs.
            return
        x, y, xys, xy_idxes = self._build_grid_sample()

        percent_queries = [0.25, 0.5, 0.75]
        idxes = [int(p * len(self.full_input_sequence)) for p in percent_queries]

        for percent_query, idx in zip(percent_queries, idxes):
            query_grid_result = self._query_grid_sample(xys, idx)
            self._process_grid_results(
                logger, prefix, percent_query, epoch_idx, query_grid_result, x, y, xy_idxes
            )


class GigachadNSFOptimizationLoop(MiniBatchOptimizationLoop):

    def __init__(
        self,
        speed_threshold: float,
        pc_target_type: PointCloudTargetType | str,
        pc_loss_type: (
            PointCloudLossType | str
        ) = PointCloudLossType.TRUNCATED_KD_TREE_FORWARD_BACKWARD,
        *args,
        **kwargs,
    ):
        super().__init__(model_class=self._model_class(), *args, **kwargs)
        self.speed_threshold = speed_threshold
        self.pc_target_type = PointCloudTargetType(pc_target_type)
        self.pc_loss_type = PointCloudLossType(pc_loss_type)

    def _model_class(self) -> type[BaseOptimizationModel]:
        return GigachadNSFModel

    def _model_constructor_args(
        self, full_input_sequence: TorchFullFrameInputSequence
    ) -> dict[str, any]:
        return super()._model_constructor_args(full_input_sequence) | dict(
            speed_threshold=self.speed_threshold,
            pc_target_type=self.pc_target_type,
            pc_loss_type=self.pc_loss_type,
        )


class GigachadNSFSincOptimizationLoop(GigachadNSFOptimizationLoop):

    def _model_constructor_args(
        self, full_input_sequence: TorchFullFrameInputSequence
    ) -> dict[str, any]:
        return super()._model_constructor_args(full_input_sequence) | dict(
            model=GigaChadFlowMLP(act_fn=ActivationFn.SINC)
        )


class GigachadNSFGaussianOptimizationLoop(GigachadNSFOptimizationLoop):

    def _model_constructor_args(
        self, full_input_sequence: TorchFullFrameInputSequence
    ) -> dict[str, any]:
        return super()._model_constructor_args(full_input_sequence) | dict(
            model=GigaChadFlowMLP(act_fn=ActivationFn.GAUSSIAN)
        )


class GigachadNSFFourtierOptimizationLoop(GigachadNSFOptimizationLoop):

    def _model_constructor_args(
        self, full_input_sequence: TorchFullFrameInputSequence
    ) -> dict[str, any]:
        return super()._model_constructor_args(full_input_sequence) | dict(
            model=GigaChadFlowMLP(encoder=FourierTemporalEmbedding())
        )


class GigachadNSFDepth10OptimizationLoop(GigachadNSFOptimizationLoop):

    def _model_constructor_args(
        self, full_input_sequence: TorchFullFrameInputSequence
    ) -> dict[str, any]:
        return super()._model_constructor_args(full_input_sequence) | dict(
            model=GigaChadFlowMLP(num_layers=10)
        )


class GigachadNSFDepth6OptimizationLoop(GigachadNSFOptimizationLoop):

    def _model_constructor_args(
        self, full_input_sequence: TorchFullFrameInputSequence
    ) -> dict[str, any]:
        return super()._model_constructor_args(full_input_sequence) | dict(
            model=GigaChadFlowMLP(num_layers=6)
        )


class GigachadNSFDepth4OptimizationLoop(GigachadNSFOptimizationLoop):

    def _model_constructor_args(
        self, full_input_sequence: TorchFullFrameInputSequence
    ) -> dict[str, any]:
        return super()._model_constructor_args(full_input_sequence) | dict(
            model=GigaChadFlowMLP(num_layers=4)
        )


class GigachadNSFDepth2OptimizationLoop(GigachadNSFOptimizationLoop):

    def _model_constructor_args(
        self, full_input_sequence: TorchFullFrameInputSequence
    ) -> dict[str, any]:
        return super()._model_constructor_args(full_input_sequence) | dict(
            model=GigaChadFlowMLP(num_layers=2)
        )
