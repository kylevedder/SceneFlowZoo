import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from models.embedders import HardEmbedder
from models.backbones import FeaturePyramidNetwork


class ConvWithNorms(nn.Module):

    def __init__(self, in_num_channels: int, out_num_channels: int,
                 kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.Conv2d(in_num_channels, out_num_channels, kernel_size,
                              stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_num_channels)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_res = self.conv(x)
        if conv_res.shape[2] == 1 and conv_res.shape[3] == 1:
            # This is a hack to get around the fact that batchnorm doesn't support
            # 1x1 convolutions
            batchnorm_res = conv_res
        else:
            batchnorm_res = self.batchnorm(conv_res)
        return self.gelu(batchnorm_res)


class ShearMatrixRegression(nn.Module):

    def __init__(self, pseudoimage_dims: Tuple[int, int],
                 pseudoimage_channels: int):
        super().__init__()
        self.pseudoimage_x = pseudoimage_dims[0]
        self.pseudoimage_y = pseudoimage_dims[1]
        self.num_channels = pseudoimage_channels

        # Scale, shear value, binary classification of upper or lower off diag shear
        out_channels = 3

        assert self.pseudoimage_x > 0, f"pseudoimage_x must be > 0, got {self.pseudoimage_x}"
        assert self.pseudoimage_y > 0, f"pseudoimage_y must be > 0, got {self.pseudoimage_y}"

        pseudoimage_stepdowns = np.ceil(
            np.max(np.log2(np.array([self.pseudoimage_x,
                                     self.pseudoimage_y])))).astype(int)
        print(f"pseudoimage_stepdowns: {pseudoimage_stepdowns}")

        conv_list = [
            ConvWithNorms(pseudoimage_channels, out_channels, 3, 1, 1)
        ]
        for _ in range(pseudoimage_stepdowns - 1):
            conv_list.append(ConvWithNorms(out_channels, out_channels, 3, 2,
                                           1))
        # Standard convolution
        conv_list.append(nn.Conv2d(out_channels, 3, 3, 2, 1))

        self.ops = nn.Sequential(*conv_list)

    def forward(self, pseudoimage: torch.Tensor) -> torch.Tensor:
        return self.ops(pseudoimage)


class PretrainEmbedding(nn.Module):

    def __init__(self, batch_size: int, device: str, VOXEL_SIZE,
                 PSEUDO_IMAGE_DIMS, POINT_CLOUD_RANGE, MAX_POINTS_PER_VOXEL,
                 FEATURE_CHANNELS, FILTERS_PER_BLOCK, PYRAMID_LAYERS) -> None:
        super().__init__()
        self.PSEUDO_IMAGE_DIMS = PSEUDO_IMAGE_DIMS
        self.batch_size = batch_size
        self.device = device
        self.embedder = HardEmbedder(voxel_size=VOXEL_SIZE,
                                 pseudo_image_dims=PSEUDO_IMAGE_DIMS,
                                 point_cloud_range=POINT_CLOUD_RANGE,
                                 max_points_per_voxel=MAX_POINTS_PER_VOXEL,
                                 feat_channels=FEATURE_CHANNELS)

        self.pyramid = FeaturePyramidNetwork(
            pseudoimage_dims=PSEUDO_IMAGE_DIMS,
            input_num_channels=FEATURE_CHANNELS + 3,
            num_filters_per_block=FILTERS_PER_BLOCK,
            num_layers_of_pyramid=PYRAMID_LAYERS)

        self.head = ShearMatrixRegression(
            PSEUDO_IMAGE_DIMS, (FEATURE_CHANNELS + 3) * (PYRAMID_LAYERS + 1))

    def forward(self, points, translations):
        shears = []
        for batch_idx in range(self.batch_size):
            embeddings = self.embedder(points[batch_idx])

            translation = translations[batch_idx]

            translation_x_latent = torch.ones(
                (1, 1, self.PSEUDO_IMAGE_DIMS[0],
                 self.PSEUDO_IMAGE_DIMS[1])).to(
                     self._get_device()) * translation[0]
            translation_y_latent = torch.ones(
                (1, 1, self.PSEUDO_IMAGE_DIMS[0],
                 self.PSEUDO_IMAGE_DIMS[1])).to(
                     self._get_device()) * translation[1]
            translation_z_latent = torch.ones(
                (1, 1, self.PSEUDO_IMAGE_DIMS[0],
                 self.PSEUDO_IMAGE_DIMS[1])).to(
                     self._get_device()) * translation[2]
            embeddings = torch.cat(
                (embeddings, translation_x_latent, translation_y_latent,
                 translation_z_latent),
                dim=1)

            pseudoimage = self.pyramid(embeddings)
            shear = self.head(pseudoimage)
            shears.append(shear)
        return torch.vstack(shears)

    def _get_device(self):
        return self.device