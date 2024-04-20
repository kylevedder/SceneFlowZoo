from dataclasses import dataclass
import torch.nn as nn
from dataloaders import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence
from .nsfp_raw_mlp import NSFPRawMLP, ActivationFn
from models.optimization.cost_functions import (
    DistanceTransform,
    DistanceTransformLossProblem,
    BaseCostProblem,
    AdditiveCosts,
    SpeedRegularizer,
)

from pointclouds import to_fixed_array_torch

from bucketed_scene_flow_eval.interfaces import LoaderType
import torch
from .base_neural_rep import BaseNeuralRep


class GigaChadRawMLP(NSFPRawMLP):

    def __init__(
        self,
        input_dim: int = 4,
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


def _make_time_feature(idx: int) -> torch.Tensor:
    return torch.tensor([idx], dtype=torch.float32)


def _make_input_feature(pc: torch.Tensor, idx: int) -> torch.Tensor:
    time_feature = _make_time_feature(idx)  # 1x1
    pc_additional_dim = time_feature.repeat(pc.shape[0], 1)
    # Concatenate into an Nx4 tensor
    concatenated_pc = torch.cat(
        [pc, pc_additional_dim.to(pc.device)],
        dim=-1,
    )

    assert (
        concatenated_pc.shape[0] == pc.shape[0]
    ), f"Expected {pc.shape[0]}, but got {concatenated_pc.shape[0]}"
    assert concatenated_pc.shape[1] == 4, f"Expected 4, but got {concatenated_pc.shape[1]}"

    return concatenated_pc


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

    def get_input_feature(self, idx: int) -> torch.Tensor:
        """
        Returns the input feature for the given index.
        """
        pc = self.get_global_pc(idx)  # N x 3
        return _make_input_feature(pc, idx)


class GigaChadNSF(BaseNeuralRep):

    def __init__(self, input_sequence: BucketedSceneFlowInputSequence):
        super().__init__()
        self.model = GigaChadRawMLP()
        self.dts = self._build_dts(input_sequence)

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

    def optim_forward_single(
        self, input_sequence: BucketedSceneFlowInputSequence
    ) -> BaseCostProblem:
        rep = self._preprocess(input_sequence)

        cost_problems: list[BaseCostProblem] = []

        for input_idx in range(len(rep) - 1):
            input_pc = rep.get_global_pc(input_idx)
            input_features = rep.get_input_feature(input_idx)
            target_dt = self.dts[input_idx + 1]

            input_flow: torch.Tensor = self.model(input_features)

            warped_pc0_points = input_pc + input_flow

            cost_problems.append(
                DistanceTransformLossProblem(
                    dt=target_dt,
                    pc=warped_pc0_points,
                )
            )
            cost_problems.append(
                SpeedRegularizer(
                    flow=input_flow.squeeze(0),
                    speed_threshold=self.speed_threshold,
                )
            )

        return AdditiveCosts(cost_problems)

    def _get_query_idx_tensors(
        self, rep: GigaChadNSFPreprocessedInput, query_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_flow_pc = self.model(rep.get_input_feature(query_idx)).squeeze(0)

        full_global_pc = rep.get_full_global_pc(query_idx)
        full_global_flow_pc = torch.zeros_like(full_global_pc)
        full_pc_mask = rep.get_full_pc_mask(query_idx)
        full_global_flow_pc[full_pc_mask] = global_flow_pc

        ego_flow = self.global_to_ego_flow(
            full_global_pc, full_global_flow_pc, rep.ego_to_globals[query_idx]
        )

        return ego_flow, full_pc_mask

    def forward_single_causal(
        self, input_sequence: BucketedSceneFlowInputSequence
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
        self, input_sequence: BucketedSceneFlowInputSequence
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
        self, input_sequence: BucketedSceneFlowInputSequence
    ) -> BucketedSceneFlowOutputSequence:

        match input_sequence.loader_type:
            case LoaderType.CAUSAL:
                return self.forward_single_causal(input_sequence)
            case LoaderType.NON_CAUSAL:
                return self.forward_single_noncausal(input_sequence)
            case _:
                raise ValueError(f"Unknown loader type: {input_sequence.loader_type}")
