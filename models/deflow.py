"""
Copied with modification from: https://github.com/KTH-RPL/DeFlow
"""

import torch
from .fast_flow_3d import FastFlow3D, FastFlow3DHeadType, FastFlow3DBackboneType


class DeFlow(FastFlow3D):
    def __init__(
        self,
        VOXEL_SIZE=[0.2, 0.2, 6],
        PSEUDO_IMAGE_DIMS=[512, 512],
        POINT_CLOUD_RANGE=[-51.2, -51.2, -3, 51.2, 51.2, 3],
        FEATURE_CHANNELS=32,
        SEQUENCE_LENGTH=2,
        bottleneck_head: FastFlow3DHeadType = FastFlow3DHeadType.DEFLOW_GRU,
        backbone: FastFlow3DBackboneType = FastFlow3DBackboneType.UNET,
    ) -> None:
        super().__init__(
            VOXEL_SIZE=VOXEL_SIZE,
            PSEUDO_IMAGE_DIMS=PSEUDO_IMAGE_DIMS,
            POINT_CLOUD_RANGE=POINT_CLOUD_RANGE,
            FEATURE_CHANNELS=FEATURE_CHANNELS,
            SEQUENCE_LENGTH=SEQUENCE_LENGTH,
            bottleneck_head=bottleneck_head,
            backbone=backbone,
        )

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {k[len("model.") :]: v for k, v in ckpt.items() if k.startswith("model.")}
        print("\nLoading... model weight from: ", ckpt_path, "\n")
        return self.load_state_dict(state_dict=state_dict, strict=False)
