"""A ConvGRU based Decoder head, as used in DeFlow.

From: https://github.com/KTH-RPL/DeFlow/blob/master/scripts/network/models/basic/decoder.py#L121
"""

import torch
import torch.nn as nn
from typing import List, Dict

# from https://github.com/weiyithu/PV-RAFT/blob/main/model/update.py
class ConvGRU(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv1d(input_dim+hidden_dim, hidden_dim, 1)
        self.convr = nn.Conv1d(input_dim+hidden_dim, hidden_dim, 1)
        self.convq = nn.Conv1d(input_dim+hidden_dim, hidden_dim, 1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        rh_x = torch.cat([r*h, x], dim=1)
        q = torch.tanh(self.convq(rh_x))

        h = (1 - z) * h + z * q
        return h

class ConvGRUDecoder(nn.Module):

    def __init__(self, pseudoimage_channels: int = 64, num_iters: int = 4):
        super().__init__()

        self.offset_encoder = nn.Linear(3, 64)

        # FIXME: not sure which one should be hidden or input?
        self.gru = ConvGRU(input_dim=64, hidden_dim=pseudoimage_channels*2)

        self.decoder = nn.Sequential(
            nn.Linear(pseudoimage_channels*3, 32), nn.GELU(),
            nn.Linear(32, 3))
        self.num_iters = num_iters

    def forward_single(self, before_pseudoimage: torch.Tensor,
                       after_pseudoimage: torch.Tensor,
                       point_offsets: torch.Tensor,
                       voxel_coords: torch.Tensor) -> torch.Tensor:
        voxel_coords = voxel_coords.long()
        # assert (voxel_coords[:, 0] == 0).all(), "Z index must be 0"

        # Voxel coords are Z, Y, X, and the pseudoimage is Channel, Y, X
        # I have confirmed via visualization that these coordinates are correct.
        after_voxel_vectors = after_pseudoimage[:, voxel_coords[:, 1],
                                                voxel_coords[:, 2]].T
        before_voxel_vectors = before_pseudoimage[:, voxel_coords[:, 1],
                                                  voxel_coords[:, 2]].T

        # [N, 64] [N, 64] -> [N, 128]
        concatenated_vectors = torch.cat([before_voxel_vectors, after_voxel_vectors], dim=1)

        # [N, 3] -> [N, 64]
        point_offsets_feature = self.offset_encoder(point_offsets)

        # [N, 128] -> [N, 128, 1]
        concatenated_vectors = concatenated_vectors.unsqueeze(2)

        # FIXME: how many iters?
        for itr in range(self.num_iters):
            concatenated_vectors = self.gru(concatenated_vectors, point_offsets_feature.unsqueeze(2))

        flow = self.decoder(torch.cat([concatenated_vectors.squeeze(2), point_offsets_feature], dim=1))
        return flow

    def forward(
            self, before_pseudoimages: torch.Tensor,
            after_pseudoimages: torch.Tensor,
            voxelizer_infos: List[Dict[str,
                                       torch.Tensor]]) -> List[torch.Tensor]:

        flow_results = []
        for before_pseudoimage, after_pseudoimage, voxelizer_info in zip(
                before_pseudoimages, after_pseudoimages, voxelizer_infos):
            point_offsets = voxelizer_info["point_offsets"]
            voxel_coords = voxelizer_info["voxel_coords"]
            flow = self.forward_single(before_pseudoimage, after_pseudoimage,
                                       point_offsets, voxel_coords)
            flow_results.append(flow)
        return flow_results