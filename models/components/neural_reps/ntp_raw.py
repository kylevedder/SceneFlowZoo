# From https://github.com/kavisha725/MBNSF/blob/main/utils/ntp_utils.py
# Functions for NTP - CVPR'2022 (https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Neural_Prior_for_Trajectory_Estimation_CVPR_2022_paper.pdf)

import torch
import torch.nn as nn
import numpy as np
import logging

from .nsfp_raw_mlp import NSFPRawMLP, ActivationFn


def load_pretrained_traj_field(pth_file, traj_len, options):
    net = NeuralTrajField(
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


class VelocityTrajDecoder(nn.Module):
    def __init__(self, traj_len):
        super().__init__()
        self.traj_rep_dim = (traj_len - 1) * 3
        self.traj_len = traj_len

    def forward(self, traj_rep, t, do_fwd_flow=False, do_bwd_flow=False, do_full_traj=False):
        traj_rep = traj_rep.view(-1, self.traj_len - 1, 3)
        result = {}
        if do_bwd_flow:
            if t == 0:
                result["flow_bwd"] = torch.zeros(
                    traj_rep.shape[0], 3, dtype=traj_rep.dtype, device=traj_rep.device
                )
            else:
                result["flow_bwd"] = -traj_rep[:, t - 1, :]

        if do_fwd_flow:
            if t == self.traj_len - 1:
                result["flow_fwd"] = torch.zeros(
                    traj_rep.shape[0], 3, dtype=traj_rep.dtype, device=traj_rep.device
                )
            else:
                result["flow_fwd"] = traj_rep[:, t, :]

        if do_full_traj:
            cumulative_traj = torch.cumsum(traj_rep, 1)
            result["traj"] = torch.cat(
                [
                    torch.zeros(
                        traj_rep.shape[0], 1, 3, dtype=traj_rep.dtype, device=traj_rep.device
                    ),
                    cumulative_traj,
                ],
                1,
            )

        return result

    def transform_pts(self, flow, pts):
        return pts + flow

    def extract_flow(self, t0, t1, traj):
        return traj[:, t1, :] - traj[:, t0, :]


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
        t = torch.full(
            size=(1,),
            fill_value=(t + 0.5) / self.traj_len,
            device=x.device,
            dtype=x.dtype,
        )
        t_embed = cosine_embed(t, self.n_freq, freq_sample_method="log")
        t_embed_t = t_embed.view(1, self.n_freq).repeat(x.shape[0], 1)
        return torch.cat([x, t_embed_t], -1)


class NeuralTrajField(nn.Module):
    def __init__(
        self,
        traj_len,
        filter_size=128,
        act_fn="relu",
    ):
        super().__init__()
        self.traj_len = traj_len

        self.embed_func = FourierSpatialTemporalEmbedding(
            traj_len=self.traj_len, n_freq=int(1 + np.ceil(np.log2(self.traj_len)))
        )

        self.input_dim = self.embed_func.embedding_dim
        self.traj_decoder = VelocityTrajDecoder(traj_len=self.traj_len)

        self.output_dim = self.traj_decoder.traj_rep_dim
        self.neural_field = NSFPRawMLP(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            latent_dim=filter_size,
            act_fn=ActivationFn(act_fn),
        )

    def forward(self, x, t, do_fwd_flow=False, do_bwd_flow=False, do_full_traj=False):
        xt_embed = self.embed_func(x, t)
        traj_rep = self.neural_field(xt_embed)
        result = self.traj_decoder(traj_rep, t, do_fwd_flow, do_bwd_flow, do_full_traj)
        result["traj_rep"] = traj_rep
        return result

    def transform_pts(self, flow, pts):
        return self.traj_decoder.transform_pts(flow, pts)

    def traj_to_pts(self, traj, pts):
        pts_traj = self.traj_decoder.transform_pts(
            traj.contiguous().view(-1, *traj.shape[2:]),
            pts.unsqueeze(1).repeat(1, self.traj_len, 1).view(-1, 3),
        )
        return pts_traj.view(-1, self.traj_len, 3)

    def extract_flow(self, t0, t1, traj):
        return self.traj_decoder.extract_flow(t0, t1, traj)

    def compute_direct_traj_consist_loss(self, traj1, traj2):
        return ((traj1 - traj2) ** 2).mean()

    def compute_traj_consist_loss(self, traj1, traj2, ref_pts1, ref_pts2, t1, t2, loss_type):
        # debug_ref_pts1 = ref_pts1.unsqueeze(-2)
        # debug_traj_sample1 = traj1[:, t1:t1+1, :]
        # debug_idmap1 = (ref_pts1.unsqueeze(-2) -  traj1[:, t1:t1+1, :])
        # debug_idmap2 = (ref_pts2.unsqueeze(-2) -  traj1[:, t2:t2+1, :])
        # debug_is_same = torch.allclose(debug_idmap1, debug_idmap2, equal_nan=True)
        traj1 = traj1 + (ref_pts1.unsqueeze(-2) - traj1[:, t1 : t1 + 1, :])
        traj2 = traj2 + (ref_pts2.unsqueeze(-2) - traj1[:, t2 : t2 + 1, :])
        if loss_type == "velocity":
            traj1_rep = traj1[:, 1:, :] - traj1[:, :-1, :]
            traj2_rep = traj2[:, 1:, :] - traj2[:, :-1, :]
        else:
            traj1_rep = traj1 - traj1.mean(-2, keepdims=True)
            traj2_rep = traj2 - traj2.mean(-2, keepdims=True)

        return ((traj1_rep - traj2_rep) ** 2).mean()
