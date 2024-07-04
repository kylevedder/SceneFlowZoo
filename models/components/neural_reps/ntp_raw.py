# From https://github.com/kavisha725/MBNSF/blob/main/utils/ntp_utils.py
# Functions for NTP - CVPR'2022 (https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Neural_Prior_for_Trajectory_Estimation_CVPR_2022_paper.pdf)

import torch
import torch.nn as nn
import numpy as np
import logging
from dataclasses import dataclass

from .nsfp_raw_mlp import NSFPRawMLP, ActivationFn


def load_pretrained_traj_field(pth_file, traj_len, options):
    net = NeuralTrajectoryField(
        traj_len=traj_len,
        filter_size=options.hidden_units,
        act_fn=options.act_fn,
        traj_type=options.traj_type,
        st_embed_type=options.st_embed_type,
    )
    net.load_state_dict(torch.load(pth_file))
    return net


def get_traj_field(pc, ref_id, traj_field):
    # Only computes the traj.
    pc = torch.from_numpy(pc).cuda()
    with torch.no_grad():
        traj_rt = traj_field(pc, ref_id, False, False, True)
    return traj_rt["traj"]


@dataclass
class DecodedTrajectory:
    global_positions: torch.Tensor
    t: int

    def get_next_position(self):
        return self.global_positions[:, self.t + 1, :]

    def get_previous_position(self):
        return self.global_positions[:, self.t - 1, :]


class VelocityTrajectoryDecoder(nn.Module):
    def __init__(self, traj_len):
        super().__init__()
        self.traj_rep_dim = (traj_len - 1) * 3
        self.traj_len = traj_len

    def forward(
        self, global_pc: torch.Tensor, global_forward_deltas: torch.Tensor, t: int
    ) -> DecodedTrajectory:
        """
        Decode the global forward deltas, which are relative to the global_pc, into global positions.

        The forward deltas from before t need to be inverted to be relative to the global_pc at t.
        """

        # Shape is N x traj_deltas x 3
        # These are the forward deltas from 0 to N - 1
        assert 0 <= t < self.traj_len, f"t must be in [0, {self.traj_len}), but got {t}"

        global_before_deltas = global_forward_deltas[:, :t, :]
        global_before_positions = (
            torch.cumsum(-global_before_deltas.flip(1), 1) + global_pc.unsqueeze(1)
        ).flip(1)

        global_after_deltas = global_forward_deltas[:, t:, :]
        global_after_positions = torch.cumsum(global_after_deltas, 1) + global_pc.unsqueeze(1)

        global_positions = torch.cat(
            [global_before_positions, global_pc.unsqueeze(1), global_after_positions], 1
        )
        return DecodedTrajectory(global_positions, t)


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


class TrajectoryNeuralRep(NSFPRawMLP):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int,
        traj_len: int,
        act_fn: ActivationFn = ActivationFn.RELU,
        num_layers: int = 8,
    ):
        super().__init__(input_dim, output_dim, latent_dim, act_fn, num_layers)
        self.trajectory_len = traj_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape the base output to be a trajectory representation
        return super().forward(x).view(-1, self.trajectory_len - 1, 3)


def cosine_embed(x, num_freq, freq_sample_method="log", scale=1):
    if freq_sample_method == "uniform":
        freq_bands = torch.linspace(1, num_freq, num_freq, device=x.device) * np.pi
    elif freq_sample_method == "log":
        freq_bands = (2 ** torch.linspace(0, num_freq - 1, num_freq, device=x.device)) * np.pi
    elif freq_sample_method == "random":
        freq_bands = torch.rand(num_freq, device=x.device) * np.pi

    return torch.cos(x[..., None] * (freq_bands[None, :]) * scale)


class FourierSpatialTemporalEmbedding(BaseSpatialTemporalEmbedding):
    def __init__(self, traj_len, n_freq):
        super().__init__()
        self.embedding_dim = n_freq + 3
        self.n_freq = n_freq
        self.traj_len = traj_len

    def forward(self, x: torch.Tensor, t: int):
        assert isinstance(x, torch.Tensor), f"x must be a tensor, but got {x} of type {type(x)}."
        assert x.shape[1] == 3, f"x must have 3 channels, but got {x.shape[1]}."
        assert isinstance(t, int), f"t must be an integer, but got {t} of type {type(t)}."
        t_torch = torch.full(
            size=(1,),
            fill_value=(t + 0.5) / self.traj_len,
            device=x.device,
            dtype=x.dtype,
        )
        t_embed = cosine_embed(t_torch, self.n_freq, freq_sample_method="log")
        t_embed_t = t_embed.view(1, self.n_freq).repeat(x.shape[0], 1)
        return torch.cat([x, t_embed_t], -1)


class NeuralTrajectoryField(nn.Module):
    def __init__(
        self,
        traj_len: int,
        filter_size: int = 128,
    ):
        super().__init__()
        self.traj_len = traj_len

        self.embed_func = FourierSpatialTemporalEmbedding(
            traj_len=self.traj_len, n_freq=int(1 + np.ceil(np.log2(self.traj_len)))
        )

        self.input_dim = self.embed_func.embedding_dim
        self.traj_decoder = VelocityTrajectoryDecoder(traj_len=self.traj_len)

        self.output_dim = self.traj_decoder.traj_rep_dim
        self.neural_field = TrajectoryNeuralRep(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            latent_dim=filter_size,
            traj_len=self.traj_len,
        )

    def forward(self, global_pc: torch.Tensor, t: int) -> DecodedTrajectory:
        xt_embed = self.embed_func(global_pc, t)
        traj_rep = self.neural_field(xt_embed)
        result = self.traj_decoder(global_pc, traj_rep, t)
        return result

    # def compute_traj_consist_loss(self, traj1, traj2, ref_pts1, ref_pts2, t1, t2, loss_type):
    #     # debug_ref_pts1 = ref_pts1.unsqueeze(-2)
    #     # debug_traj_sample1 = traj1[:, t1:t1+1, :]
    #     # debug_idmap1 = (ref_pts1.unsqueeze(-2) -  traj1[:, t1:t1+1, :])
    #     # debug_idmap2 = (ref_pts2.unsqueeze(-2) -  traj1[:, t2:t2+1, :])
    #     # debug_is_same = torch.allclose(debug_idmap1, debug_idmap2, equal_nan=True)
    #     traj1 = traj1 + (ref_pts1.unsqueeze(-2) - traj1[:, t1 : t1 + 1, :])
    #     traj2 = traj2 + (ref_pts2.unsqueeze(-2) - traj1[:, t2 : t2 + 1, :])
    #     if loss_type == "velocity":
    #         traj1_rep = traj1[:, 1:, :] - traj1[:, :-1, :]
    #         traj2_rep = traj2[:, 1:, :] - traj2[:, :-1, :]
    #     else:
    #         traj1_rep = traj1 - traj1.mean(-2, keepdims=True)
    #         traj2_rep = traj2 - traj2.mean(-2, keepdims=True)

    #     return ((traj1_rep - traj2_rep) ** 2).mean()
