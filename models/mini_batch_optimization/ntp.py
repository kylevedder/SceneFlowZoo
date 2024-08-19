from pytorch_lightning.loggers.logger import Logger
from .mini_batch_optim_loop import MiniBatchOptimizationLoop, MinibatchedSceneFlowInputSequence
from models.components.neural_reps import fNT, DecodedTrajectories
from dataloaders import TorchFullFrameInputSequence, TorchFullFrameOutputSequence
from dataclasses import dataclass
from models import BaseOptimizationModel
from models.components.optimization.cost_functions import (
    BaseCostProblem,
    PassthroughCostProblem,
    DistanceTransform,
    TruncatedForwardKDTreeLossProblem,
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
from bucketed_scene_flow_eval.datastructures import O3DVisualizer, PointCloud


# @dataclass
# class NTPTrajectoryConsistencyProblem(BaseCostProblem):
#     before_trajectory: DecodedTrajectory
#     query_trajectory: DecodedTrajectory
#     after_trajectory: DecodedTrajectory

#     def base_cost(self) -> torch.Tensor:
#         """
#         All trajectories are in global coordinates, and we are just computing the mean of the L2 squared distance between the positions.
#         """

#         before_query_diff = (
#             self.before_trajectory.global_positions - self.query_trajectory.global_positions
#         ) ** 2
#         after_query_diff = (
#             self.after_trajectory.global_positions - self.query_trajectory.global_positions
#         ) ** 2

#         return before_query_diff.mean() + after_query_diff.mean()


@dataclass
class NTPTrajectoryConsistencyProblem(BaseCostProblem):
    t1: DecodedTrajectories
    t2: DecodedTrajectories

    def base_cost(self) -> torch.Tensor:
        """
        All trajectories are in global coordinates, and we are just computing the mean of the L2 squared distance between the positions.
        """

        assert (
            self.t1.global_positions.shape[1] > self.t2.global_positions.shape[1]
        ), "Expected t1 to be longer than t2"

        assert (
            self.t2.global_positions.shape[1] > 0
        ), f"Expected t2 to be longer than 0; got {self.t2.global_positions.shape[1]}. Full shapes are {self.t1.global_positions.shape}, {self.t2.global_positions.shape}"

        t_diff = self.t2.t - self.t1.t

        t2_positions = self.t2.global_positions
        t1_positions = self.t1.global_positions[:, t_diff:]

        assert (
            t1_positions.shape == t2_positions.shape
        ), f"Expected {t1_positions.shape}, but got {t2_positions.shape}. Times are {self.t1.t}, {self.t2.t} and input shapes are {self.t1.global_positions.shape}, {self.t2.global_positions.shape}"

        position_diff = torch.linalg.vector_norm(t1_positions - t2_positions, dim=-1)
        return position_diff.mean()


@dataclass
class NTPPreprocessedInput:
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


class NTPModel(BaseOptimizationModel):
    def __init__(
        self,
        full_input_sequence: TorchFullFrameInputSequence,
        consistency_loss_weight: float = 1.0,
    ) -> None:
        super().__init__(full_input_sequence)
        print(f"full_input_sequence: {len(full_input_sequence)}")
        self.model = fNT(
            traj_len=len(full_input_sequence),
        )
        self.consistency_loss_weight = consistency_loss_weight

    def _preprocess(self, input_sequence: TorchFullFrameInputSequence) -> NTPPreprocessedInput:

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

        return NTPPreprocessedInput(
            full_global_pcs=full_global_pcs,
            full_global_pcs_mask=full_global_pcs_mask,
            full_global_auxillary_pcs=full_global_camera_pcs,
            ego_to_globals=ego_to_globals,
            sequence_idxes=sequence_idxes,
            sequence_total_length=sequence_total_length,
        )

    def _make_trajectory_consistency_losses(
        self,
        query_trajectory: DecodedTrajectories,
        after_trajectory: DecodedTrajectories,
    ) -> BaseCostProblem:
        return AdditiveCosts(
            [
                NTPTrajectoryConsistencyProblem(query_trajectory, after_trajectory),
            ]
        )

    def _make_neighbor_chamfer_loss(
        self,
        query_trajectory: DecodedTrajectories,
        gt_after_pc: torch.Tensor,
    ) -> BaseCostProblem:
        est_after_pc = query_trajectory.get_next_position()

        return AdditiveCosts(
            [
                TruncatedChamferLossProblem(
                    est_after_pc, gt_after_pc, distance_type=ChamferDistanceType.BOTH_DIRECTION
                ),
            ]
        )

    def optim_forward_single(
        self, input_sequence: TorchFullFrameInputSequence, logger: Logger
    ) -> BaseCostProblem:
        rep = self._preprocess(input_sequence)
        # Query index is 1 to n-2 because we need to preserve there being and additional before and after point.

        cost_problems: list[BaseCostProblem] = []
        for local_query_index in range(1, len(rep) - 1):
            local_after_idx = local_query_index + 1

            gt_query_pc = rep.get_global_lidar_pc(local_query_index)

            # Base trajectory
            query_trajectory: DecodedTrajectories = self.model(
                gt_query_pc,
                rep.sequence_idxes[local_query_index],
            )

            # Standard Chamfer based scene flow loss for clouds.
            cost_problems.append(
                self._make_neighbor_chamfer_loss(
                    query_trajectory,
                    rep.get_global_lidar_pc(local_after_idx),
                )
            )

            est_after_pc = query_trajectory.get_next_position()

            # Trajectories of the one step look ahead
            after_trajectory: DecodedTrajectories = self.model(
                est_after_pc,
                rep.sequence_idxes[local_after_idx],
            )

            # Neighbors should have the same trajectories
            cost_problems.append(
                self._make_trajectory_consistency_losses(query_trajectory, after_trajectory)
            )

        return AdditiveCosts(cost_problems)

    def _compute_ego_flow(
        self,
        rep: NTPPreprocessedInput,
        query_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        query_pc = rep.get_global_lidar_pc(query_idx)
        query_trajectory: DecodedTrajectories = self.model(
            query_pc,
            rep.sequence_idxes[query_idx],
        )

        # We need to construct the full global pc, by using the query trajectory to get the next position
        full_global_pc = rep.get_full_global_pc(query_idx)
        full_global_flow_pc = full_global_pc.clone()
        full_pc_mask = rep.get_full_pc_mask(query_idx)
        full_global_flow_pc[full_pc_mask] = query_trajectory.get_next_position()

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
        return self._forward_single_noncausal(input_sequence, logger)


class NTPOptimizationLoop(MiniBatchOptimizationLoop):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(model_class=NTPModel, *args, **kwargs)
