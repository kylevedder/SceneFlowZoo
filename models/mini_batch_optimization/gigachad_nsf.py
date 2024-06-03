from pytorch_lightning.loggers.logger import Logger
from .mini_batch_optim_loop import MiniBatchOptimizationLoop, MinibatchedSceneFlowInputSequence
from models.components.neural_reps import GigaChadRawMLP
from dataloaders import TorchFullFrameInputSequence, TorchFullFrameOutputSequence
from dataclasses import dataclass
from models import BaseOptimizationModel
from models.components.optimization.cost_functions import (
    BaseCostProblem,
    DistanceTransform,
    AdditiveCosts,
    PointwiseLossProblem,
    DistanceTransformLossProblem,
    TruncatedChamferLossProblem,
    SpeedRegularizer,
)
from bucketed_scene_flow_eval.interfaces import LoaderType
from pointclouds import to_fixed_array_torch
import enum
import torch
import torch.nn as nn
import tqdm


class LossTypeEnum(enum.Enum):
    TRUNCATED_CHAMFER = 0
    DISTANCE_TRANSFORM = 1


class QueryDirection(enum.Enum):
    FORWARD = 1
    REVERSE = -1


class ChamferTargetType(enum.Enum):
    LIDAR = "lidar"
    LIDAR_CAMERA = "lidar_camera"


@dataclass
class GlobalNormalizer:
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    @staticmethod
    def from_input_sequence(input_sequence: TorchFullFrameInputSequence) -> "GlobalNormalizer":
        min_x, min_y, min_z = float("inf"), float("inf"), float("inf")
        max_x, max_y, max_z = float("-inf"), float("-inf"), float("-inf")

        for idx in range(len(input_sequence)):
            pc = input_sequence.get_global_pc(idx)
            min_x = min(min_x, torch.min(pc[:, 0]).item())
            min_y = min(min_y, torch.min(pc[:, 1]).item())
            min_z = min(min_z, torch.min(pc[:, 2]).item())

            max_x = max(max_x, torch.max(pc[:, 0]).item())
            max_y = max(max_y, torch.max(pc[:, 1]).item())
            max_z = max(max_z, torch.max(pc[:, 2]).item())

        return GlobalNormalizer(min_x, min_y, min_z, max_x, max_y, max_z)

    def normalize(self, pc: torch.Tensor) -> torch.Tensor:
        """
        Normalize the pc so that it will be centered at 0, 0, 0

        Importantly, the relative distances of the different axses are preserved
        """
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]

        # Calculate center coordinates
        center_x = (self.max_x + self.min_x) / 2
        center_y = (self.max_y + self.min_y) / 2
        center_z = (self.max_z + self.min_z) / 2

        # Shift to the origin (without in-place operations)
        x = x - center_x
        y = y - center_y
        z = z - center_z

        # Find maximum extent
        max_extent = torch.max(
            torch.tensor(
                [self.max_x - self.min_x, self.max_y - self.min_y, self.max_z - self.min_z]
            )
        )

        # Scale to normalize (without in-place operations)
        x = x / max_extent
        y = y / max_extent
        z = z / max_extent

        return torch.stack([x, y, z], dim=1)


def _make_time_feature(idx: int, total_entries: int, device: torch.device) -> torch.Tensor:
    # Make the time feature zero mean
    if total_entries <= 1:
        # Handle divide by zero
        return torch.tensor([0.0], dtype=torch.float32, device=device)
    max_idx = total_entries - 1
    return torch.tensor([(idx / max_idx) - 0.5], dtype=torch.float32, device=device)


def _make_input_feature(
    pc: torch.Tensor,
    idx: int,
    total_entries: int,
    query_direction: QueryDirection,
    normalizer: GlobalNormalizer,
) -> torch.Tensor:
    assert pc.shape[1] == 3, f"Expected 3, but got {pc.shape[1]}"
    assert pc.dim() == 2, f"Expected 2, but got {pc.dim()}"
    assert isinstance(
        query_direction, QueryDirection
    ), f"Expected QueryDirection, but got {query_direction}"

    time_feature = _make_time_feature(idx, total_entries, pc.device)  # 1x1

    direction_feature = torch.tensor(
        [query_direction.value], dtype=torch.float32, device=pc.device
    )  # 1x1
    pc_time_dim = time_feature.repeat(pc.shape[0], 1).contiguous()
    pc_direction_dim = direction_feature.repeat(pc.shape[0], 1).contiguous()

    normalized_pc = pc  # normalizer.normalize(pc)

    # Concatenate into a feature tensor
    return torch.cat(
        [normalized_pc, pc_time_dim, pc_direction_dim],
        dim=-1,
    )


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

    def get_global_lidar_auxillary_pc(self, idx: int) -> torch.Tensor:
        lidar_pc = self.full_global_pcs[idx]
        lidar_mask = self.full_global_pcs_mask[idx]
        valid_pc = lidar_pc[lidar_mask]

        # If the camera pc is not available, return the lidar pc
        auxillary_pc = self.full_global_auxillary_pcs[idx]
        assert auxillary_pc is not None, "Expected auxillary_pc to be available"
        # concatenate the camera pc to the lidar pc
        valid_pc = torch.cat([valid_pc, auxillary_pc], dim=0)

        return valid_pc.clone().detach().requires_grad_(True)

    def get_global_lidar_pc(self, idx: int) -> torch.Tensor:
        lidar_pc = self.full_global_pcs[idx]
        lidar_mask = self.full_global_pcs_mask[idx]
        valid_lidar_pc = lidar_pc[lidar_mask]
        return valid_lidar_pc.clone().detach().requires_grad_(True)

    def get_full_pc_mask(self, idx: int) -> torch.Tensor:
        mask = self.full_global_pcs_mask[idx]
        # Conver to bool tensor
        return mask.bool()


class GigachadNSFModel(BaseOptimizationModel):

    def __init__(
        self,
        full_input_sequence: TorchFullFrameInputSequence,
        speed_threshold: float,
        chamfer_target_type: ChamferTargetType | str,
    ) -> None:
        super().__init__(full_input_sequence)
        self.model = GigaChadRawMLP()
        self.global_normalizer = GlobalNormalizer.from_input_sequence(full_input_sequence)
        self.speed_threshold = speed_threshold
        if isinstance(chamfer_target_type, str):
            chamfer_target_type = ChamferTargetType(chamfer_target_type)
        self.chamfer_target_type = chamfer_target_type

        self._prep_neural_prior(self.model)

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

    def _cycle_consistency(self, rep: GigaChadNSFPreprocessedInput) -> BaseCostProblem:
        cost_problems: list[BaseCostProblem] = []
        for idx in range(len(rep) - 1):
            pc = rep.get_global_lidar_pc(idx)
            sequence_idx = rep.sequence_idxes[idx]
            sequence_total_length = rep.sequence_total_length
            base_input_features = _make_input_feature(
                pc,
                sequence_idx,
                sequence_total_length,
                QueryDirection.FORWARD,
                self.global_normalizer,
            )
            forward_flow: torch.Tensor = self.model(base_input_features)
            forward_flowed_pc = pc + forward_flow

            forward_flowed_input_features = _make_input_feature(
                forward_flowed_pc,
                sequence_idx + 1,
                sequence_total_length,
                QueryDirection.REVERSE,
                self.global_normalizer,
            )

            backward_flow: torch.Tensor = self.model(forward_flowed_input_features)
            round_trip_flowed_pc = forward_flowed_pc + backward_flow

            cost_problems.append(PointwiseLossProblem(pred=round_trip_flowed_pc, target=pc))
        return AdditiveCosts(cost_problems)

    def _k_step_cost(
        self,
        rep: GigaChadNSFPreprocessedInput,
        k: int,
        query_direction: QueryDirection,
        loss_type: LossTypeEnum = LossTypeEnum.TRUNCATED_CHAMFER,
        speed_limit: float | None = None,
    ) -> BaseCostProblem:
        assert k >= 1, f"Expected k >= 1, but got {k}"

        def process_subk(
            anchor_pc: torch.Tensor, anchor_idx: int, subk: int, query_direction: QueryDirection
        ) -> tuple[BaseCostProblem, torch.Tensor]:
            sequence_idx = rep.sequence_idxes[anchor_idx]
            features = _make_input_feature(
                anchor_pc,
                sequence_idx,
                rep.sequence_total_length,
                query_direction,
                self.global_normalizer,
            )
            flow: torch.Tensor = self.model(features)
            anchor_pc = anchor_pc + flow

            index_offset = (subk + 1) * query_direction.value

            match self.chamfer_target_type:
                case ChamferTargetType.LIDAR:
                    target_pc = rep.get_global_lidar_pc(anchor_idx + index_offset)
                case ChamferTargetType.LIDAR_CAMERA:
                    target_pc = rep.get_global_lidar_auxillary_pc(anchor_idx + index_offset)

            match loss_type:
                case LossTypeEnum.TRUNCATED_CHAMFER:
                    problem: BaseCostProblem = TruncatedChamferLossProblem(
                        warped_pc=anchor_pc,
                        target_pc=target_pc,
                    )
                case LossTypeEnum.DISTANCE_TRANSFORM:
                    raise NotImplementedError("Distance transform not implemented")

            if speed_limit is not None:
                problem = AdditiveCosts(
                    [
                        problem,
                        SpeedRegularizer(
                            flow=flow.squeeze(0),
                            speed_threshold=speed_limit,
                        ),
                    ]
                )

            return problem, anchor_pc

        costs = []

        if query_direction == QueryDirection.FORWARD:
            anchor_idx_range = range(len(rep) - k)
        elif query_direction == QueryDirection.REVERSE:
            anchor_idx_range = range(k, len(rep))

        for anchor_idx in anchor_idx_range:
            anchor_pc = rep.get_global_lidar_pc(anchor_idx)

            for subk in range(k):
                problem, anchor_pc = process_subk(anchor_pc, anchor_idx, subk, query_direction)
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
        return AdditiveCosts(
            [
                self._k_step_cost(rep, 1, QueryDirection.FORWARD, speed_limit=self.speed_threshold),
                self._k_step_cost(rep, 1, QueryDirection.REVERSE, speed_limit=self.speed_threshold),
                self._k_step_cost(rep, 3, QueryDirection.FORWARD),
                self._k_step_cost(rep, 3, QueryDirection.REVERSE),
                self._cycle_consistency(rep) * 0.01,
            ]
        )

    def _compute_ego_flow(
        self,
        rep: GigaChadNSFPreprocessedInput,
        query_idx: int,
        direction: QueryDirection = QueryDirection.FORWARD,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_features = _make_input_feature(
            rep.get_global_lidar_pc(query_idx),
            query_idx,
            len(rep),
            direction,
            self.global_normalizer,
        )
        global_flow_pc = self.model(input_features).squeeze(0)

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


class GigachadNSFOptimizationLoop(MiniBatchOptimizationLoop):
    def __init__(
        self, speed_threshold: float, chamfer_target_type: ChamferTargetType | str, *args, **kwargs
    ):
        super().__init__(model_class=GigachadNSFModel, *args, **kwargs)
        self.speed_threshold = speed_threshold
        if isinstance(chamfer_target_type, str):
            chamfer_target_type = ChamferTargetType(chamfer_target_type)
        self.chamfer_target_type = chamfer_target_type

    def _model_constructor_args(
        self, full_input_sequence: TorchFullFrameInputSequence
    ) -> dict[str, any]:
        return super()._model_constructor_args(full_input_sequence) | dict(
            speed_threshold=self.speed_threshold, chamfer_target_type=self.chamfer_target_type
        )
