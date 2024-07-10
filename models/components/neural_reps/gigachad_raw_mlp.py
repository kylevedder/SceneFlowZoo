from torch._tensor import Tensor
from .nsfp_raw_mlp import NSFPRawMLP, ActivationFn
from dataclasses import dataclass
import torch
import enum


class QueryDirection(enum.Enum):
    FORWARD = 1
    REVERSE = -1


@dataclass
class ModelFlowResult:
    flow: torch.Tensor  # N x 3


@dataclass
class ModelOccFlowResult(ModelFlowResult):
    occ: torch.Tensor  # N


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

    normalized_pc = pc

    # Concatenate into a feature tensor
    return torch.cat(
        [normalized_pc, pc_time_dim, pc_direction_dim],
        dim=-1,
    )


class GigaChadFlowMLP(NSFPRawMLP):

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

    def forward(
        self,
        pc: torch.Tensor,
        idx: int,
        total_entries: int,
        query_direction: QueryDirection,
    ) -> ModelFlowResult:
        input_feature = _make_input_feature(pc, idx, total_entries, query_direction)
        res = super().forward(input_feature)
        return ModelFlowResult(flow=res)


class GigaChadOccFlowMLP(NSFPRawMLP):

    def __init__(
        self,
        input_dim: int = 5,
        output_dim: int = 4,
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

    def forward(
        self,
        pc: torch.Tensor,
        idx: int,
        total_entries: int,
        query_direction: QueryDirection,
    ) -> ModelOccFlowResult:
        input_feature = _make_input_feature(pc, idx, total_entries, query_direction)
        res = super().forward(input_feature)
        assert res.shape[1] == 4, f"Expected 4, but got {res.shape[1]}"
        return ModelOccFlowResult(flow=res[:, :3], occ=res[:, 3])
