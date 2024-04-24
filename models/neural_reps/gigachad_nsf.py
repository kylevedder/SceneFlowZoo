from dataclasses import dataclass
import torch.nn as nn
from dataloaders import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence
from .nsfp_raw_mlp import NSFPRawMLP, ActivationFn
from models.optimization.cost_functions import (
    DistanceTransform,
    DistanceTransformLossProblem,
    PointwiseLossProblem,
    BaseCostProblem,
    AdditiveCosts,
    SpeedRegularizer,
    TruncatedChamferLossProblem,
)
from models.optimization.utils import EarlyStopping

from pointclouds import to_fixed_array_torch

from bucketed_scene_flow_eval.interfaces import LoaderType
import torch
from .base_neural_rep import BaseNeuralRep
from pytorch_lightning.loggers import Logger
import enum
from typing import Optional


class LossTypeEnum(enum.Enum):
    TRUNCATED_CHAMFER = 0
    DISTANCE_TRANSFORM = 1


class QueryDirection(enum.Enum):
    FORWARD = 1
    REVERSE = -1


@dataclass
class GlobalNormalizer:
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    @staticmethod
    def from_input_sequence(input_sequence: BucketedSceneFlowInputSequence) -> "GlobalNormalizer":
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


class GigaChadRawMLP(NSFPRawMLP):

    def __init__(
        self,
        input_dim: int = 5,
        output_dim: int = 3,
        latent_dim: int = 128,
        act_fn: ActivationFn = ActivationFn.RELU,
        num_layers: int = 8,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            act_fn=act_fn,
            num_layers=num_layers,
        )


def _make_time_feature(idx: int, total_entries: int) -> torch.Tensor:
    # Make the time feature zero mean
    if total_entries <= 1:
        # Handle divide by zero
        return torch.tensor([0.0], dtype=torch.float32)
    max_idx = total_entries - 1
    return torch.tensor([(idx / max_idx) - 0.5], dtype=torch.float32)


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

    time_feature = _make_time_feature(idx, total_entries)  # 1x1

    direction_feature = torch.tensor([query_direction.value], dtype=torch.float32)  # 1x1
    pc_time_dim = time_feature.repeat(pc.shape[0], 1)
    pc_direction_dim = direction_feature.repeat(pc.shape[0], 1)

    normalized_pc = pc  # normalizer.normalize(pc)

    # Concatenate into a feature tensor
    return torch.cat(
        [normalized_pc, pc_time_dim.to(pc.device), pc_direction_dim.to(pc.device)],
        dim=-1,
    )


@dataclass
class GigaChadNSFPreprocessedInput:
    full_global_pcs: list[torch.Tensor]
    full_global_pcs_mask: list[torch.Tensor]
    ego_to_globals: list[torch.Tensor]

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

    def get_global_pc(self, idx: int) -> torch.Tensor:
        pc = self.full_global_pcs[idx]
        mask = self.full_global_pcs_mask[idx]
        return pc[mask].clone().detach().requires_grad_(True)

    def get_full_pc_mask(self, idx: int) -> torch.Tensor:
        mask = self.full_global_pcs_mask[idx]
        # Conver to bool tensor
        return mask.bool()


class GigaChadNSF(BaseNeuralRep):

    def __init__(self, input_sequence: BucketedSceneFlowInputSequence, speed_threshold: float):
        super().__init__()
        self.model = GigaChadRawMLP()
        self.dts = self._build_dts(input_sequence)
        for dt in self.dts:
            print("DT", dt)
        self.global_normalizer = GlobalNormalizer.from_input_sequence(input_sequence)
        self.speed_threshold = speed_threshold

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

    def _build_dts(self, input_sequence: BucketedSceneFlowInputSequence) -> list[DistanceTransform]:
        preprocess_result = self._preprocess(input_sequence.clone().detach())
        return [
            DistanceTransform.from_pointcloud(preprocess_result.get_global_pc(idx).clone().detach())
            for idx in range(len(preprocess_result))
        ]

    def _preprocess(
        self, input_sequence: BucketedSceneFlowInputSequence
    ) -> GigaChadNSFPreprocessedInput:

        full_global_pcs, full_global_pcs_mask, ego_to_globals = [], [], []

        for idx in range(len(input_sequence)):
            full_pc, full_pc_mask = input_sequence.get_full_global_pc(
                idx
            ), input_sequence.get_full_pc_mask(idx)

            ego_to_global = input_sequence.pc_poses_ego_to_global[idx]

            full_global_pcs.append(full_pc)
            full_global_pcs_mask.append(full_pc_mask)
            ego_to_globals.append(ego_to_global)

        return GigaChadNSFPreprocessedInput(
            full_global_pcs=full_global_pcs,
            full_global_pcs_mask=full_global_pcs_mask,
            ego_to_globals=ego_to_globals,
        )

    def _cycle_consistency(self, rep: GigaChadNSFPreprocessedInput) -> BaseCostProblem:
        cost_problems: list[BaseCostProblem] = []
        for idx in range(len(rep) - 1):
            pc = rep.get_global_pc(idx)
            base_input_features = _make_input_feature(
                pc, idx, len(rep), QueryDirection.FORWARD, self.global_normalizer
            )
            forward_flow: torch.Tensor = self.model(base_input_features)
            forward_flowed_pc = pc + forward_flow

            forward_flowed_input_features = _make_input_feature(
                forward_flowed_pc,
                idx + 1,
                len(rep),
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
        loss_type: LossTypeEnum = LossTypeEnum.DISTANCE_TRANSFORM,
        speed_limit: Optional[float] = None,
    ) -> BaseCostProblem:
        assert k >= 1, f"Expected k >= 1, but got {k}"

        def process_subk(
            pc: torch.Tensor, subk: int, query_direction: QueryDirection
        ) -> tuple[BaseCostProblem, torch.Tensor]:
            features = _make_input_feature(
                pc, input_idx + subk, len(rep), query_direction, self.global_normalizer
            )
            flow: torch.Tensor = self.model(features)
            pc = pc + flow

            index_offset = (subk + 1) * query_direction.value

            target_pc = rep.get_global_pc(input_idx + index_offset)
            distance_transform = self.dts[input_idx + index_offset]

            if loss_type == LossTypeEnum.DISTANCE_TRANSFORM:
                problem: BaseCostProblem = DistanceTransformLossProblem(
                    dt=distance_transform,
                    pc=pc,
                )
            elif loss_type == LossTypeEnum.TRUNCATED_CHAMFER:
                problem = TruncatedChamferLossProblem(
                    warped_pc=pc,
                    target_pc=target_pc,
                )

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

            return problem, pc

        costs = []

        if query_direction == QueryDirection.FORWARD:
            input_idx_range = range(len(rep) - k)
        elif query_direction == QueryDirection.REVERSE:
            input_idx_range = range(k, len(rep))

        for input_idx in input_idx_range:
            pc = rep.get_global_pc(input_idx)
            for subk in range(k):
                problem, pc = process_subk(pc, subk, query_direction)
                costs.append(problem)
        return AdditiveCosts(costs)

    def optim_forward_single(
        self,
        input_sequence: BucketedSceneFlowInputSequence,
        optim_step: int,
        early_stopping: EarlyStopping,
        logger: Logger,
    ) -> BaseCostProblem:
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

    def _get_query_idx_tensors(
        self, rep: GigaChadNSFPreprocessedInput, query_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_features = _make_input_feature(
            rep.get_global_pc(query_idx),
            query_idx,
            len(rep),
            QueryDirection.FORWARD,
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

    def forward_single_causal(
        self, input_sequence: BucketedSceneFlowInputSequence, logger: Logger
    ) -> BucketedSceneFlowOutputSequence:
        assert (
            input_sequence.loader_type == LoaderType.CAUSAL
        ), f"Expected causal, but got {input_sequence.loader_type}"

        rep = self._preprocess(input_sequence)

        query_idx = -2

        ego_flow, full_pc_mask = self._get_query_idx_tensors(rep, query_idx)

        return BucketedSceneFlowOutputSequence(
            ego_flows=torch.unsqueeze(ego_flow, 0),
            valid_flow_mask=torch.unsqueeze(full_pc_mask, 0),
        )

    def forward_single_noncausal(
        self, input_sequence: BucketedSceneFlowInputSequence, logger: Logger
    ) -> BucketedSceneFlowOutputSequence:
        assert (
            input_sequence.loader_type == LoaderType.NON_CAUSAL
        ), f"Expected non-causal, but got {input_sequence.loader_type}"

        padded_n = input_sequence.get_pad_n()
        rep = self._preprocess(input_sequence)

        ego_flows = []
        valid_flow_masks = []

        # For each index from 0 to len - 2, get the flow
        for idx in range(len(rep) - 1):
            ego_flow, full_pc_mask = self._get_query_idx_tensors(rep, idx)

            padded_ego_flow = to_fixed_array_torch(ego_flow, padded_n)
            padded_full_pc_mask = to_fixed_array_torch(full_pc_mask, padded_n)
            ego_flows.append(padded_ego_flow)
            valid_flow_masks.append(padded_full_pc_mask)

        return BucketedSceneFlowOutputSequence(
            ego_flows=torch.stack(ego_flows),
            valid_flow_mask=torch.stack(valid_flow_masks),
        )

    def forward_single(
        self, input_sequence: BucketedSceneFlowInputSequence, logger: Logger
    ) -> BucketedSceneFlowOutputSequence:

        match input_sequence.loader_type:
            case LoaderType.CAUSAL:
                return self.forward_single_causal(input_sequence, logger)
            case LoaderType.NON_CAUSAL:
                return self.forward_single_noncausal(input_sequence, logger)
            case _:
                raise ValueError(f"Unknown loader type: {input_sequence.loader_type}")
