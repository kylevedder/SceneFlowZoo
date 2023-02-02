import torch
import torch.nn as nn
from typing import List, Tuple, Dict


class FastFlowDecoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(nn.Linear(131, 32), nn.GELU(),
                                     nn.Linear(32, 3))

    def forward_single(self, before_pseudoimage: torch.Tensor,
                       after_pseudoimage: torch.Tensor,
                       point_offsets: torch.Tensor,
                       voxel_coords: torch.Tensor) -> torch.Tensor:
        assert (voxel_coords[:, 0] == 0).all(), "Z index must be 0"

        after_voxel_vectors = after_pseudoimage[:, voxel_coords[:, 1].long(),
                                                voxel_coords[:, 2].long()].T
        before_voxel_vectors = before_pseudoimage[:, voxel_coords[:, 1].long(),
                                                  voxel_coords[:, 2].long()].T
        concatenated_vectors = torch.cat(
            [before_voxel_vectors, after_voxel_vectors, point_offsets], dim=1)

        flow = self.decoder(concatenated_vectors)
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


class ConvWithNorms(nn.Module):

    def __init__(self, in_num_channels: int, out_num_channels: int,
                 kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.Conv2d(in_num_channels, out_num_channels, kernel_size,
                              stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_num_channels)
        self.nonlinearity = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_res = self.conv(x)
        if conv_res.shape[2] == 1 and conv_res.shape[3] == 1:
            # This is a hack to get around the fact that batchnorm doesn't support
            # 1x1 convolutions
            batchnorm_res = conv_res
        else:
            batchnorm_res = self.batchnorm(conv_res)
        return self.nonlinearity(batchnorm_res)


class FastFlowDecoderStepDown(nn.Module):

    def __init__(self, num_stepdowns: int) -> None:
        super().__init__()
        assert num_stepdowns > 0, "stepdown_factor must be positive"
        self.num_stepdowns = num_stepdowns

        self.pseudoimage_stepdown_head = nn.ModuleList()
        # Build convolutional pseudoimage stepdown head
        for i in range(num_stepdowns):
            if i == 0:
                in_channels = 64
            else:
                in_channels = 16
            out_channels = 16
            self.pseudoimage_stepdown_head.append(
                ConvWithNorms(in_channels, out_channels, 3, 2, 1))
