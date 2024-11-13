from torch._tensor import Tensor
from .nsfp_raw_mlp import NSFPRawMLP, ActivationFn
from dataclasses import dataclass
import torch
import enum
from abc import ABC, abstractmethod
import typing


class QueryDirection(enum.Enum):
    FORWARD = 1
    REVERSE = -1


@dataclass
class ModelFlowResult:
    flow: torch.Tensor  # N x 3


@dataclass
class ModelOccFlowResult(ModelFlowResult):
    occ: torch.Tensor  # N


class BaseEncoder(ABC, torch.nn.Module):

    @abstractmethod
    def encode(
        self, pc: Tensor, idx: int, total_entries: int, query_direction: QueryDirection
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    def forward(self, entries: tuple[Tensor, int, int, QueryDirection]) -> Tensor:
        (pc, idx, total_entries, query_direction) = entries
        return self.encode(pc, idx, total_entries, query_direction)


class SimpleEncoder(BaseEncoder):

    def _make_time_feature(
        self, idx: int, total_entries: int, device: torch.device
    ) -> torch.Tensor:
        # Make the time feature zero mean
        if total_entries <= 1:
            # Handle divide by zero
            return torch.tensor([0.0], dtype=torch.float32, device=device)
        max_idx = total_entries - 1
        return torch.tensor([(idx / max_idx) - 0.5], dtype=torch.float32, device=device)

    def encode(
        self, pc: Tensor, idx: int, total_entries: int, query_direction: QueryDirection
    ) -> Tensor:
        assert pc.shape[1] == 3, f"Expected 3, but got {pc.shape[1]}"

        assert pc.dim() == 2, f"Expected 2, but got {pc.dim()}"
        assert isinstance(
            query_direction, QueryDirection
        ), f"Expected QueryDirection, but got {query_direction}"

        time_feature = self._make_time_feature(idx, total_entries, pc.device)  # 1x1

        direction_feature = torch.tensor(
            [query_direction.value], dtype=torch.float32, device=pc.device
        )  # 1x1
        pc_time_dim = time_feature.repeat(pc.shape[0], 1).contiguous()
        pc_direction_dim = direction_feature.repeat(pc.shape[0], 1).contiguous()

        # Concatenate into a feature tensor
        return torch.cat(
            [pc, pc_time_dim, pc_direction_dim],
            dim=-1,
        )

    def __len__(self):
        return 5  # point + time + direction


def cosine_embed(x: torch.Tensor, num_freq: int, freq_sample_method="log", scale: float = 1.0):
    if freq_sample_method == "uniform":
        freq_bands = torch.linspace(1, num_freq, num_freq, device=x.device) * torch.pi
    elif freq_sample_method == "log":
        freq_bands = (2 ** torch.linspace(0, num_freq - 1, num_freq, device=x.device)) * torch.pi
    elif freq_sample_method == "random":
        freq_bands = torch.rand(num_freq, device=x.device) * torch.pi

    return torch.cos(x[..., None] * (freq_bands[None, :]) * scale)


class FourierTemporalEmbedding(SimpleEncoder):
    def __init__(self, n_freq: int = 14):
        super().__init__()
        self.n_freq = n_freq

    def encode(
        self, pc: Tensor, idx: int, total_entries: int, query_direction: QueryDirection
    ) -> torch.Tensor:
        simple_t_encoded = super().encode(pc, idx, total_entries, query_direction)
        t_torch = torch.full(
            size=(1,),
            fill_value=(idx + 0.5) / total_entries,
            device=pc.device,
            dtype=pc.dtype,
        )
        t_embed = cosine_embed(t_torch, self.n_freq, freq_sample_method="log")
        t_embed_t = t_embed.view(1, self.n_freq).repeat(pc.shape[0], 1)
        return torch.cat([simple_t_encoded, t_embed_t], dim=-1)

    def __len__(self):
        return super().__len__() + self.n_freq


class EulerFlowMLP(NSFPRawMLP):

    def __init__(
        self,
        output_dim: int = 3,
        latent_dim: int = 128,
        act_fn: ActivationFn = ActivationFn.RELU,
        num_layers: int = 8,
        encoder: BaseEncoder = SimpleEncoder(),
    ):
        super().__init__(
            input_dim=len(encoder),
            output_dim=output_dim,
            latent_dim=latent_dim,
            act_fn=act_fn,
            num_layers=num_layers,
        )
        self.nn_layers = torch.compile(torch.nn.Sequential(encoder, self.nn_layers))

    @typing.no_type_check
    def forward(
        self,
        pc: torch.Tensor,
        idx: int,
        total_entries: int,
        query_direction: QueryDirection,
    ) -> ModelFlowResult:
        entries = (pc, idx, total_entries, query_direction)
        res = self.nn_layers(entries)
        return ModelFlowResult(flow=res)


class EulerFlowOccFlowMLP(NSFPRawMLP):

    def __init__(
        self,
        output_dim: int = 4,
        latent_dim: int = 128,
        act_fn: ActivationFn = ActivationFn.SINC,
        num_layers: int = 8,
        encoder: BaseEncoder = FourierTemporalEmbedding(),
        with_compile: bool = True,
    ):
        super().__init__(
            input_dim=len(encoder),
            output_dim=output_dim,
            latent_dim=latent_dim,
            act_fn=act_fn,
            num_layers=num_layers,
            with_compile=with_compile,
        )
        self.nn_layers = torch.nn.Sequential(encoder, self.nn_layers)
        if with_compile:
            self.nn_layers = torch.compile(self.nn_layers)

    @typing.no_type_check
    def forward(
        self,
        pc: torch.Tensor,
        idx: int,
        total_entries: int,
        query_direction: QueryDirection,
    ) -> ModelOccFlowResult:
        entries = (pc, idx, total_entries, query_direction)
        res = self.nn_layers(entries)
        assert res.shape[1] == 4, f"Expected 4, but got {res.shape[1]}"
        return ModelOccFlowResult(flow=res[:, :3], occ=res[:, 3])
