import torch
import torch.nn as nn

from model.embedders import Embedder
from model.backbones import FeaturePyramidNetwork
from model.attention import JointConvAttention
from model.heads import NeuralSceneFlowPrior, NeuralSceneFlowPriorOptimizable

class FastFlow(nn.Module):

    def __init__(self,
                 batch_size: int,
                 device: str,
                 VOXEL_SIZE=(0.14, 0.14, 4),
                 PSEUDO_IMAGE_DIMS=(512, 512),
                 POINT_CLOUD_RANGE=(-33.28, -33.28, -3, 33.28, 33.28, 1),
                 MAX_POINTS_PER_VOXEL=128,
                 FEATURE_CHANNELS=16,
                 FILTERS_PER_BLOCK=3,
                 PYRAMID_LAYERS=1,
                 SEQUENCE_LENGTH=5,
                 NSFP_FILTER_SIZE=64,
                 NSFP_NUM_LAYERS=4) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.embedder = Embedder(voxel_size=VOXEL_SIZE,
                                 pseudo_image_dims=PSEUDO_IMAGE_DIMS,
                                 point_cloud_range=POINT_CLOUD_RANGE,
                                 max_points_per_voxel=MAX_POINTS_PER_VOXEL,
                                 feat_channels=FEATURE_CHANNELS)

        self.pyramid = FeaturePyramidNetwork(
            pseudoimage_dims=PSEUDO_IMAGE_DIMS,
            input_num_channels=FEATURE_CHANNELS + 3,
            num_filters_per_block=FILTERS_PER_BLOCK,
            num_layers_of_pyramid=PYRAMID_LAYERS)