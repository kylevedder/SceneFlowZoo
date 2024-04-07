from typing import List
import time

import torch
import torch.nn as nn

from dataloaders import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence
from models.backbones import FastFlowUNet, FastFlowUNetXL
from models.embedders import DynamicEmbedder
from models.heads import FastFlowDecoder, FastFlowDecoderStepDown, ConvGRUDecoder
from pointclouds.losses import warped_pc_loss
from .base_model import BaseModel
import enum


class FastFlow3DSelfSupervisedLoss:
    def __init__(self, device: str = None):
        super().__init__()

    def _warped_loss(
        self,
        input_batch: list[BucketedSceneFlowInputSequence],
        model_res: list[BucketedSceneFlowOutputSequence],
    ):

        # Input batch length should be the same as the estimated flows
        assert len(input_batch) == len(
            model_res
        ), f"input_batch {len(input_batch)} != model_res {len(model_res)}"
        warped_loss = 0
        for input_item, output_item in zip(input_batch, model_res):

            target_pc = input_item.get_global_pc(-1)
            warped_pc = input_item.get_full_global_pc(-2) + output_item.get_full_ego_flow(0)

            warped_loss += warped_pc_loss(warped_pc, target_pc)
        return warped_loss

    def __call__(
        self,
        input_batch: list[BucketedSceneFlowInputSequence],
        model_results: List[BucketedSceneFlowOutputSequence],
    ):
        return {"loss": self._warped_loss(input_batch, model_results)}


class FastFlow3DBucketedLoaderLoss:
    def __init__(self, device: str = None, fast_mover_scale: bool = False):
        super().__init__()
        self.fast_mover_scale = fast_mover_scale

    def __call__(
        self,
        input_batch: List[BucketedSceneFlowInputSequence],
        model_results: List[BucketedSceneFlowOutputSequence],
    ):
        # Input batch length should be the same as the estimated flows
        assert len(input_batch) == len(
            model_results
        ), f"input_batch {len(input_batch)} != model_results {len(model_results)}"

        total_loss = 0
        # Iterate through the batch
        for input_item, output_item in zip(input_batch, model_results):
            assert (
                len(output_item) == 1
            ), f"Expected a single output flow, but got {len(output_item)}"

            assert len(input_item) >= 2, f"Expected at least two frames, but got {len(input_item)}"

            source_idx = len(input_item) - 2

            ############################################################
            # Mask for extracting the eval points for the gt flow
            ############################################################

            # Ensure that all entries in the gt mask are included in the input mask
            assert (
                (
                    input_item.get_full_pc_gt_flow_mask(source_idx)
                    & input_item.get_full_pc_mask(source_idx)
                )
                == input_item.get_full_pc_gt_flow_mask(source_idx)
            ).all(), "GT mask is not a subset of the input mask."

            # Valid only if valid GT flow and valid PC and valid from the voxelizer
            valid_loss_mask = (
                input_item.get_full_pc_gt_flow_mask(source_idx)
                & input_item.get_full_pc_mask(source_idx)
                & output_item.get_full_flow_mask(0)
            )

            gt_flow = input_item.get_full_ego_pc_gt_flowed(source_idx) - input_item.get_full_ego_pc(
                source_idx
            )
            est_flow = output_item.get_full_ego_flow(0)
            flow_difference = est_flow - gt_flow
            loss_difference = flow_difference[valid_loss_mask]
            diff_l2 = torch.norm(loss_difference, dim=1, p=2).mean()
            total_loss += diff_l2
        return {
            "loss": total_loss,
        }


class FastFlow3DHeadType(enum.Enum):
    LINEAR = "linear"
    STEPDOWN = "stepdown"
    DEFLOW_GRU = "deflow_gru"


class FastFlow3DBackboneType(enum.Enum):
    UNET = "unet"
    UNET_XL = "unet_xl"


class FastFlow3D(BaseModel):
    """
    FastFlow3D based on the paper:
    https://arxiv.org/abs/2103.01306v5

    Note that there are several small differences between this implementation and the paper:
     - We use a different loss function (predict flow for P_-1 to P_0 instead of P_0 to and
       unseen P_1); referred to as pc0 and pc1 in the code.
    """

    def __init__(
        self,
        VOXEL_SIZE,
        PSEUDO_IMAGE_DIMS,
        POINT_CLOUD_RANGE,
        FEATURE_CHANNELS,
        SEQUENCE_LENGTH,
        bottleneck_head: FastFlow3DHeadType = FastFlow3DHeadType.LINEAR,
        backbone: FastFlow3DBackboneType = FastFlow3DBackboneType.UNET,
    ) -> None:
        super().__init__()
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH
        assert (
            self.SEQUENCE_LENGTH == 2
        ), "This implementation only supports a sequence length of 2."
        self.embedder = DynamicEmbedder(
            voxel_size=VOXEL_SIZE,
            pseudo_image_dims=PSEUDO_IMAGE_DIMS,
            point_cloud_range=POINT_CLOUD_RANGE,
            feat_channels=FEATURE_CHANNELS,
        )

        match backbone:
            case FastFlow3DBackboneType.UNET_XL:
                self.backbone: nn.Module = FastFlowUNetXL()
            case FastFlow3DBackboneType.UNET:
                self.backbone: nn.Module = FastFlowUNet()
            case _:
                raise ValueError(f"Invalid backbone type: {backbone}")

        match bottleneck_head:
            case FastFlow3DHeadType.LINEAR:
                self.head: nn.Module = FastFlowDecoder(pseudoimage_channels=FEATURE_CHANNELS * 2)
            case FastFlow3DHeadType.STEPDOWN:
                self.head: nn.Module = FastFlowDecoderStepDown(
                    voxel_pillar_size=VOXEL_SIZE[:2], num_stepdowns=3
                )
            case FastFlow3DHeadType.DEFLOW_GRU:
                self.head: nn.Module = ConvGRUDecoder(num_iters=4)
            case _:
                raise ValueError(f"Invalid head type: {bottleneck_head}")

    def _indices_to_mask(self, points, mask_indices):
        mask = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
        assert mask_indices.max() < len(
            mask
        ), f"Mask indices go outside of the tensor range. {mask_indices.max()} >= {len(mask)}"
        mask[mask_indices] = True
        return mask

    def _transform_pc(self, pc: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        """
        Transform an Nx3 point cloud by a 4x4 transformation matrix.
        """

        homogenious_pc = torch.cat((pc, torch.ones((pc.shape[0], 1), device=pc.device)), dim=1)
        return torch.matmul(transform, homogenious_pc.T).T[:, :3]

    def _model_forward(
        self,
        full_pc0s: list[tuple[torch.Tensor, torch.Tensor]],
        full_pc1s: list[tuple[torch.Tensor, torch.Tensor]],
        pc0_transforms: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[BucketedSceneFlowOutputSequence]:
        """
        Args:
            pc0s: A list (len=batch size) of point source point clouds.
            pc1s: A list (len=batch size) of point target point clouds.
        Returns:
            A list of BucketedSceneFlowOutputItem dataclasses (of length batch size).
        """

        masked_pc0s = [full_pc[mask] for full_pc, mask in full_pc0s]
        masked_pc1s = [full_pc[mask] for full_pc, mask in full_pc1s]
        masked_pc0_before_pseudoimages, masked_pc0_voxel_infos_lst = self.embedder(masked_pc0s)
        masked_pc1_before_pseudoimages, masked_pc1_voxel_infos_lst = self.embedder(masked_pc1s)
        # breakpoint()

        grid_flow_pseudoimage = self.backbone(
            masked_pc0_before_pseudoimages, masked_pc1_before_pseudoimages
        )
        masked_valid_flows = self.head(
            torch.cat((masked_pc0_before_pseudoimages, masked_pc1_before_pseudoimages), dim=1),
            grid_flow_pseudoimage,
            masked_pc0_voxel_infos_lst,
        )

        batch_output = []

        for (
            (full_p0, full_p0_mask),
            masked_p0,
            (pc0_sensor_to_ego, pc0_ego_to_global),
            masked_p0_voxel_info,
            masked_valid_flow,
        ) in zip(
            full_pc0s,
            masked_pc0s,
            pc0_transforms,
            masked_pc0_voxel_infos_lst,
            masked_valid_flows,
        ):
            masked_pc0_valid_point_mask = self._indices_to_mask(
                masked_p0, masked_p0_voxel_info["point_idxes"]
            )

            # The voxelizer provides valid points for the masked point clouds.
            masked_flow = torch.zeros_like(masked_p0)
            masked_flow[masked_pc0_valid_point_mask] = masked_valid_flow
            # Results must be backed out to the full point cloud.
            full_flow = torch.zeros_like(full_p0)
            full_flow[full_p0_mask] = masked_flow

            # Updated mask for the full point cloud.
            full_flow_mask = full_p0_mask.clone()
            full_flow_mask[full_p0_mask] = masked_pc0_valid_point_mask

            ego_flow = self.global_to_ego_flow(full_p0, full_flow, pc0_ego_to_global)

            batch_output.append(
                BucketedSceneFlowOutputSequence(
                    ego_flows=torch.unsqueeze(ego_flow, 0),
                    valid_flow_mask=torch.unsqueeze(full_flow_mask, 0),
                )
            )
        return batch_output

    def forward(
        self, batched_sequence: List[BucketedSceneFlowInputSequence]
    ) -> List[BucketedSceneFlowOutputSequence]:
        """
        Args:
            batched_sequence: A list (len=batch size) of BucketedSceneFlowItems.

        Returns:
            A list (len=batch size) of BucketedSceneFlowOutputItem.
        """
        pc0s = [(e.get_full_global_pc(-2), e.get_full_pc_mask(-2)) for e in batched_sequence]
        pc1s = [(e.get_full_global_pc(-1), e.get_full_pc_mask(-1)) for e in batched_sequence]
        pc0_transforms = [e.get_pc_transform_matrices(-2) for e in batched_sequence]
        res = self._model_forward(pc0s, pc1s, pc0_transforms)
        return res
