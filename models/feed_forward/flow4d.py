"""
Copied with modification from: https://github.com/dgist-cvlab/Flow4D
"""


import torch
import dztimer
from models import BaseModel, ForwardMode
from typing import Any, List
from dataloaders import (
    BucketedSceneFlowInputSequence,
    BucketedSceneFlowOutputSequence,
)
from pytorch_lightning.loggers import Logger

class Flow4D(BaseModel):
    def __init__(self, voxel_size = [0.2, 0.2, 0.2],
                 point_cloud_range = [-51.2, -51.2, -2.2, 51.2, 51.2, 4.2],
                 grid_feature_size = [512, 512, 32],
                 num_frames = 5):
        super().__init__()

        point_output_ch = 16
        voxel_output_ch = 16

        self.num_frames = num_frames
        print('voxel_size = {}, pseudo_dims = {}, input_num_frames = {}'.format(voxel_size, grid_feature_size, self.num_frames))

        # TODO: Port 4D embedder, backbone network etc.
        # self.embedder_4D = DynamicEmbedder_4D(voxel_size=voxel_size,
        #                                 pseudo_image_dims=[grid_feature_size[0], grid_feature_size[1], grid_feature_size[2], num_frames], 
        #                                 point_cloud_range=point_cloud_range,
        #                                 feat_channels=point_output_ch)
        
        # self.network_4D = Network_4D(in_channel=point_output_ch, out_channel=voxel_output_ch)
        # self.seperate_feat = Seperate_to_3D(num_frames)
        # self.pointhead_3D = Point_head(voxel_feat_dim=voxel_output_ch, point_feat_dim=point_output_ch)

        self.timer = dztimer.Timing()
        self.timer.start("Total")

    def forward(
        self,
        forward_mode: ForwardMode,
        batched_sequence: List[BucketedSceneFlowInputSequence],
        logger: Logger,
    ) -> List[BucketedSceneFlowOutputSequence]:
        """
        Args:
            batched_sequence: A list (len=batch size) of BucketedSceneFlowItems.

        Returns:
            A list (len=batch size) of BucketedSceneFlowOutputItem.
        """
        raise NotImplementedError

    def loss_fn(
        self, 
        input_batch: List[BucketedSceneFlowInputSequence],
        model_res: List[BucketedSceneFlowOutputSequence],
    ) -> dict[str, torch.Tensor]:

        raise NotImplementedError

    def _model_forward(
        self, batched_sequence: List[BucketedSceneFlowInputSequence]
    ) -> list[BucketedSceneFlowOutputSequence]:
        """
        input: using the batch from dataloader, which is a dict
               Detail: [pc0, pc1, pose0, pose1]
        output: the predicted flow, pose_flow, and the valid point index of pc0
        """

        batch_sizes = len(batched_sequence)

        pose_flows = []
        transform_pc0s = []
        transform_pc_m_frames = []

        # TODO: Transform pcs to pc[-1] frame, get pose_flows=transform_pc0 - origin_pc0
        transform_pc0s = [(e.get_full_target_pc(-2), e.get_full_pc_mask(-2)) for e in batched_sequence]
        if self.num_frames > 2: 
            for i in range(self.num_frames-2):
                transform_pc_mi = [(e.get_full_target_pc(i), e.get_full_pc_mask(i)) for e in batched_sequence]
                transform_pc_m_frames.append(transform_pc_mi)
        pc1s = [e.get_full_ego_pc(-1) for e in batched_sequence]

        # TODO: embedder_4D, network_4D, Seperate_to_3D, pointhead_3D

        # TODO: the return type should be List[BucketedSceneFlowOutputSequence]

        raise NotImplementedError