# From https://github.com/kavisha725/MBNSF/blob/main/utils/ntp_utils.py
# Functions for NTP - CVPR'2022 (https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Neural_Prior_for_Trajectory_Estimation_CVPR_2022_paper.pdf)

import torch
import torch.nn as nn
import numpy as np
import logging
from dataclasses import dataclass

from .nsfp_raw_mlp import NSFPRawMLP, ActivationFn

from bucketed_scene_flow_eval.datastructures import O3DVisualizer, PointCloud

K_NUM_TRAJECTORIES = 256


class BaseSpatialTemporalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 4

    def forward(self, x, t):
        """
        x : -1, 3
        t : scalar
        """
        return torch.cat([x, torch.full((x.shape[0], 1), t, device=x.device, dtype=x.dtype)], -1)


class FourierSpatialTemporalEmbedding(nn.Module):
    def __init__(self, traj_len, n_freq):
        super().__init__()
        self.embedding_dim = n_freq
        self.n_freq = n_freq
        self.traj_len = traj_len

    def cosine_embed(
        self, x: torch.Tensor, num_freq: int, freq_sample_method="log", scale: float = 1
    ):
        if freq_sample_method == "uniform":
            freq_bands = torch.linspace(1, num_freq, num_freq, device=x.device) * np.pi
        elif freq_sample_method == "log":
            freq_bands = (2 ** torch.linspace(0, num_freq - 1, num_freq, device=x.device)) * np.pi
        elif freq_sample_method == "random":
            freq_bands = torch.rand(num_freq, device=x.device) * np.pi

        return torch.cos(x[..., None] * (freq_bands[None, :]) * scale)

    def forward(self, t: torch.Tensor):
        assert isinstance(t, torch.Tensor), f"x must be a tensor, but got {t} of type {type(t)}."
        assert t.ndim == 1, f"t must have 1 dimension, but got {t.ndim}."

        normalized_t = (t + 0.5) / self.traj_len
        return self.cosine_embed(normalized_t, self.n_freq, freq_sample_method="log")


@dataclass
class TrajectoryBasis:
    global_positions: torch.Tensor
    t: int

    def __post__init__(self):
        assert (
            self.global_positions.ndim == 4
        ), f"global_positions must have 4 dimensions (N, num_basis, traj_len, 3), but got {self.global_positions.ndim}."
        assert (
            self.global_positions.shape[3] == 3
        ), f"global_positions must have 3 channels, but got {self.global_positions.shape[3]}."

        assert (
            self.global_positions.shape[2] > 0
        ), f"traj_len must be greater than 0, but got {self.global_positions.shape[2]}. Full shape is {self.global_positions.shape}."


@dataclass
class DecodedTrajectories:
    global_positions: torch.Tensor
    t: int

    def __post_init__(self):
        assert (
            self.global_positions.ndim == 3
        ), f"global_positions must have 3 dimensions (N, traj_len, 3), but got {self.global_positions.ndim}."
        assert (
            self.global_positions.shape[2] == 3
        ), f"global_positions must have 3 channels, but got {self.global_positions.shape[2]}."
        assert (
            self.global_positions.shape[1] > 0
        ), f"traj_len must be greater than 0, but got {self.global_positions.shape[1]}. Full shape is {self.global_positions.shape}."

    @staticmethod
    def from_basis(
        basis: TrajectoryBasis, linear_combination: torch.Tensor
    ) -> "DecodedTrajectories":
        assert isinstance(
            basis, TrajectoryBasis
        ), f"basis must be an instance of TrajectoryBasis, but got {basis} of type {type(basis)}."
        assert isinstance(
            linear_combination, torch.Tensor
        ), f"linear_combination must be a tensor, but got {linear_combination} of type {type(linear_combination)}."
        assert (
            linear_combination.ndim == 2
        ), f"linear_combination must have 2 dimensions(N, num_basis), but got {linear_combination.ndim}."

        assert basis.global_positions.shape[2] > 0, (
            f"traj_len must be greater than 0, but got {basis.global_positions.shape[2]}. "
            f"Overall shape is {basis.global_positions.shape}."
        )

        # Take the linear combination of the basis positions to get the global positions.
        # Basis basis.global_positions is N x num_basis x traj_len x 3
        # Linear combination is N x num_basis
        # Resulting shape is N x traj_len x 3
        assert linear_combination.shape[1] == basis.global_positions.shape[1], (
            f"linear_combination must have the same number of basis as basis.global_positions, "
            f"but got {linear_combination.shape[1]} and {basis.global_positions.shape[1]}. "
            f"Overall shapes are {linear_combination.shape} and {basis.global_positions.shape}."
        )
        global_positions = torch.einsum(
            "nb, nbtp -> ntp", linear_combination, basis.global_positions
        )
        assert (
            global_positions.shape[1] > 0
        ), f"traj_len must be greater than 0, but got {global_positions.shape[1]}. Einsum input shapes are {linear_combination.shape} and {basis.global_positions.shape}."
        return DecodedTrajectories(global_positions, basis.t)

    def get_next_position(self) -> torch.Tensor:
        return self.global_positions[:, 1, :]


class fphi(NSFPRawMLP):
    """
    Maps global position and time to a low dimensional bottleneck.
    """

    def __init__(
        self,
        positition_time_dim: int = 4,
        low_dimensional_bottleneck: int = 4,
        latent_dim: int = 128,
        act_fn: ActivationFn = ActivationFn.RELU,
        num_layers: int = 4,
        with_compile: bool = True,
    ):
        super().__init__(
            positition_time_dim,
            low_dimensional_bottleneck,
            latent_dim,
            act_fn,
            num_layers,
            with_compile,
        )


class fa(NSFPRawMLP):
    """
    Maps low dimensional bottleneck to linear combination over basis positions.
    """

    def __init__(
        self,
        low_dimensional_bottleneck: int = 4,
        num_trajectories_k: int = K_NUM_TRAJECTORIES,
        latent_dim: int = 128,
        act_fn: ActivationFn = ActivationFn.RELU,
        num_layers: int = 4,
        with_compile: bool = True,
    ):
        super().__init__(
            low_dimensional_bottleneck,
            num_trajectories_k,
            latent_dim,
            act_fn,
            num_layers,
            with_compile,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : N x 4
        """
        raw_combination = super().forward(x)
        normalized_linear_combination = torch.softmax(raw_combination, dim=1)
        return normalized_linear_combination


class ft(nn.Module):
    """
    Maps timestep t to K trajectory bases.
    """

    def __init__(
        self,
        num_trajectories_k: int = K_NUM_TRAJECTORIES,
        traj_len: int = 20,
        latent_dim: int = 128,
        act_fn: ActivationFn = ActivationFn.RELU,
        num_layers: int = 4,
        with_compile: bool = True,
    ):
        super().__init__()
        self.num_trajectories_k = num_trajectories_k
        self.traj_len = traj_len
        self.embedder = FourierSpatialTemporalEmbedding(
            traj_len=self.traj_len, n_freq=int(1 + np.ceil(np.log2(self.traj_len)))
        )
        self.mlp = NSFPRawMLP(
            self.embedder.embedding_dim,
            num_trajectories_k * 3 * (self.traj_len - 1),
            latent_dim,
            act_fn,
            num_layers,
            with_compile,
        )

    def forward(
        self, positions_tensor: torch.Tensor, t: int, times_tensor: torch.Tensor
    ) -> TrajectoryBasis:
        """
        p : N,

        Returns:
        Trajectory positions N x 3
        """
        assert isinstance(
            positions_tensor, torch.Tensor
        ), f"positions_tensor must be a tensor, but got {positions_tensor} of type {type(positions_tensor)}."
        assert (
            positions_tensor.ndim == 2
        ), f"positions_tensor must have 2 dimensions, but got {positions_tensor.ndim}."
        assert (
            positions_tensor.shape[1] == 3
        ), f"positions_tensor must have 3 channels, but got {positions_tensor.shape[1]}."
        assert isinstance(
            times_tensor, torch.Tensor
        ), f"times_tensor must be a tensor, but got {times_tensor} of type {type(times_tensor)}."
        assert times_tensor.ndim == 1, f"times must have 1 dimension, but got {times_tensor.ndim}."

        assert isinstance(t, int), f"t must be an integer, but got {t} of type {type(t)}."
        embedded_time = self.embedder(times_tensor)

        # These deltas are changes from 0 to N - 1
        # As per the paper, the rolled out trajectory is only forward looking, so we don't need to
        # compute the global positions for the reverse deltas.

        full_trajectory_velocity_space_relative_deltas: torch.Tensor = self.mlp(embedded_time).view(
            -1, self.num_trajectories_k, (self.traj_len - 1), 3
        )

        assert 0 <= t < self.traj_len, f"t must be in [0, {self.traj_len}), but got {t}"

        future_relative_deltas = full_trajectory_velocity_space_relative_deltas[:, :, t:]
        future_relative_positions = torch.cumsum(future_relative_deltas, 2)
        # add zero position to the start of the trajectory
        future_relative_positions = torch.cat(
            [
                torch.zeros_like(full_trajectory_velocity_space_relative_deltas[:, :, :1]),
                future_relative_positions,
            ],
            2,
        )

        # future_relative_positions.shape = torch.Size([N, 256, t:, 3])
        # positions_tensor.shape = torch.Size([N, 3])
        future_global_positions = future_relative_positions + positions_tensor[:, None, None, :]
        assert future_global_positions.shape[2] > 0, (
            f"traj_len must be greater than 0, but got {future_global_positions.shape[2]}. "
            f"Overall shape is {future_global_positions.shape}."
        )
        return TrajectoryBasis(future_global_positions, t)


class fNT(nn.Module):

    def __init__(self, traj_len: int = 20):
        super().__init__()
        self.trajectory_length = traj_len
        self.fphi_net = fphi()
        self.fa_net = fa()
        self.ft_net = ft()

    def _visualize(self, x: torch.Tensor, decoded_trajectories: DecodedTrajectories):
        vis = O3DVisualizer(point_size=2)
        pc_x = PointCloud(x.detach().cpu().numpy())
        vis.add_pointcloud(pc_x, color=[1, 0, 0])
        query_trajectory_matrix = decoded_trajectories.global_positions.detach().cpu().numpy()
        trajectory_pc_list = [
            PointCloud(query_trajectory_matrix[:, idx])
            for idx in range(query_trajectory_matrix.shape[1])
        ]
        for pc1, pc2 in zip(trajectory_pc_list, trajectory_pc_list[1:]):
            vis.add_lineset(pc1, pc2)
            vis.add_pointcloud(pc1, color=[0, 0, 1])

        vis.run()

    def forward(self, x: torch.Tensor, t: int) -> DecodedTrajectories:
        """
        x : N x 3 global position
        t : scalar timestep
        """
        assert x.shape[1] == 3, f"p must have 3 channels, but got {x.shape[1]}."
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        p = torch.cat([x, t_tensor.unsqueeze(-1)], -1)

        trajectory_basis = self.ft_net(x, t, t_tensor)
        bottleneck = self.fphi_net(p)
        linear_combination = self.fa_net(bottleneck)

        decoded_trajectories = DecodedTrajectories.from_basis(trajectory_basis, linear_combination)

        return decoded_trajectories
