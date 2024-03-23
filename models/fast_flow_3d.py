from typing import List
import time

import torch
import torch.nn as nn

from dataloaders import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence
from models.backbones import FastFlowUNet, FastFlowUNetXL
from models.embedders import DynamicEmbedder
from models.heads import FastFlowDecoder, FastFlowDecoderStepDown
from pointclouds.losses import warped_pc_loss


class FastFlow3DSelfSupervisedLoss:
    def __init__(self, device: str = None):
        super().__init__()

    def _warped_loss(
        self,
        input_batch: list[BucketedSceneFlowInputSequence],
        model_res: list[BucketedSceneFlowOutputSequence],
    ):
        warped_loss = 0
        for input_item, output_item in zip(input_batch, model_res):
            warped_loss += warped_pc_loss(output_item.pc0_warped_points, input_item.target_pc)
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

            ############################################################
            # Mask for extracting the eval points for the gt flow
            ############################################################

            # Ensure that all entries in the gt mask are included in the input mask
            assert (
                (input_item.raw_gt_flowed_source_pc_mask & input_item.raw_source_pc_mask)
                == input_item.raw_gt_flowed_source_pc_mask
            ).all(), f"{input_item.raw_gt_flowed_source_pc_mask} & {input_item.raw_source_pc_mask} != {input_item.raw_gt_flowed_source_pc_mask}"

            # Start with raw source pc, which includes not ground points
            raw_gt_evaluation_mask = input_item.raw_source_pc_mask.clone()
            # Add in the mask for the valid points from the voxelizer
            raw_gt_evaluation_mask[raw_gt_evaluation_mask.clone()] = (
                output_item.pc0_valid_point_mask.to(raw_gt_evaluation_mask.device)
            )
            # Add in the mask for the valid points from the gt flow.
            raw_gt_evaluation_mask = (
                raw_gt_evaluation_mask & input_item.raw_gt_flowed_source_pc_mask
            )

            ############################################################
            # Mask for extracting the eval points for the output flow
            ############################################################

            # Disable the mask for the points that are not in the voxelizer
            output_evaluation_mask = input_item.raw_gt_flowed_source_pc_mask[
                input_item.raw_source_pc_mask
            ] & output_item.pc0_valid_point_mask.to(raw_gt_evaluation_mask.device)

            source_pc = input_item.raw_source_pc[raw_gt_evaluation_mask]
            gt_flowed_pc = input_item.raw_gt_flowed_source_pc[raw_gt_evaluation_mask]
            gt_flow = gt_flowed_pc - source_pc

            est_flow = output_item.flow[output_evaluation_mask]

            assert (
                gt_flow.shape == est_flow.shape
            ), f"Flow shapes are different: gt {gt_flow.shape} != est {est_flow.shape}"

            loss_difference = est_flow - gt_flow
            diff_l2 = torch.norm(loss_difference, dim=1, p=2).mean()
            total_loss += diff_l2
        return {
            "loss": total_loss,
        }


class FastFlow3D(nn.Module):
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
        bottleneck_head=False,
        xl_backbone=False,
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
        if xl_backbone:
            self.backbone = FastFlowUNetXL()
        else:
            self.backbone = FastFlowUNet()
        if bottleneck_head:
            self.head = FastFlowDecoderStepDown(voxel_pillar_size=VOXEL_SIZE[:2], num_stepdowns=3)
        else:
            self.head = FastFlowDecoder(pseudoimage_channels=FEATURE_CHANNELS * 2)

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

    def _global_to_ego_flow(
        self,
        global_full_pc0: torch.Tensor,
        global_warped_full_pc0: torch.Tensor,
        pc0_ego_to_global: torch.Tensor,
    ) -> torch.Tensor:

        ego_full_pc0 = self._transform_pc(global_full_pc0, pc0_ego_to_global)
        ego_warped_full_pc0 = self._transform_pc(global_warped_full_pc0, pc0_ego_to_global)

        return ego_warped_full_pc0 - ego_full_pc0

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

            warped_full_pc0 = full_p0 + full_flow
            ego_flow = self._global_to_ego_flow(full_p0, warped_full_pc0, pc0_ego_to_global)

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
