import torch
import torch.nn as nn

from models.embedders import HardEmbedder, DynamicEmbedder
from models.backbones import FastFlowUNet
from models.heads import FastFlowDecoder
from pointclouds import warped_pc_loss

from typing import Dict


class FastFlow3DLoss():

    def __init__(self, device: str = None):
        super().__init__()

    def __call__(self, flows, pc0_points_lst, pc1_points_lst):
        total_loss = 0
        for flow, pc0_points, pc1_points in zip(flows, pc0_points_lst,
                                                pc1_points_lst):
            warped_pc1_points = pc0_points + flow

            loss = warped_pc_loss(pc1_points, warped_pc1_points)
            total_loss += loss
        return total_loss


class FastFlow3D(nn.Module):
    """
    FastFlow3D based on the paper:
    https://arxiv.org/abs/2103.01306v5

    Note that there are several small differences between this implementation and the paper:
     - We use a different loss function (predict flow for P_-1 to P_0 instead of P_0 to and 
       unseen P_1); referred to as pc0 and pc1 in the code.
    """

    def __init__(self, device: str, VOXEL_SIZE,
                 PSEUDO_IMAGE_DIMS, POINT_CLOUD_RANGE, MAX_POINTS_PER_VOXEL,
                 FEATURE_CHANNELS, SEQUENCE_LENGTH) -> None:
        super().__init__()
        self.device = device
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH
        assert self.SEQUENCE_LENGTH == 2, "This implementation only supports a sequence length of 2."
        self.embedder = DynamicEmbedder(voxel_size=VOXEL_SIZE,
                                        pseudo_image_dims=PSEUDO_IMAGE_DIMS,
                                        point_cloud_range=POINT_CLOUD_RANGE,
                                        feat_channels=FEATURE_CHANNELS)

        self.backbone = FastFlowUNet()
        self.head = FastFlowDecoder()

    def forward(self, batched_sequence: Dict[str, torch.Tensor]):
        pc_arrays = batched_sequence['pc_array_stack']
        pc0s = pc_arrays[:, 0]
        pc1s = pc_arrays[:, 1]
        pc0_before_pseudoimages, pc0_points_coordinates_lst = self.embedder(
            pc0s)
        pc1_before_pseudoimages, pc1_points_coordinates_lst = self.embedder(
            pc1s)

        grid_flow_pseudoimage = self.backbone(pc0_before_pseudoimages,
                                              pc1_before_pseudoimages)
        flows = self.head(
            torch.cat((pc0_before_pseudoimages, pc1_before_pseudoimages),
                      dim=1), grid_flow_pseudoimage,
            pc0_points_coordinates_lst)

        pc0_points_lst = [points for points, _ in pc0_points_coordinates_lst]
        pc1_points_lst = [points for points, _ in pc1_points_coordinates_lst]
        return flows, pc0_points_lst, pc1_points_lst
