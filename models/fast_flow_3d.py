from typing import List

import torch
import torch.nn as nn

from dataloaders import BucketedSceneFlowItem, BucketedSceneFlowOutputItem
from models.backbones import FastFlowUNet, FastFlowUNetXL
from models.embedders import DynamicEmbedder
from models.heads import FastFlowDecoder, FastFlowDecoderStepDown
from pointclouds.losses import warped_pc_loss


class FastFlow3DSelfSupervisedLoss:
    def __init__(self, device: str = None):
        super().__init__()

    def _warped_loss(self, input_batch: list[BucketedSceneFlowItem], model_res: list[BucketedSceneFlowOutputItem]):
        warped_loss = 0
        for input_item, output_item in zip(input_batch, model_res):
            warped_loss += warped_pc_loss(output_item.pc0_warped_points, input_item.target_pc)
        return warped_loss

    def __call__(self, input_batch: list[BucketedSceneFlowItem], model_results: List[BucketedSceneFlowOutputItem]):
        return {"loss": self._warped_loss(input_batch, model_results)}


class FastFlow3DBucketedLoaderLoss:
    def __init__(self, device: str = None, fast_mover_scale: bool = False):
        super().__init__()
        self.fast_mover_scale = fast_mover_scale

    def __call__(
        self,
        input_batch: List[BucketedSceneFlowItem],
        model_results: List[BucketedSceneFlowOutputItem],
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
            assert ((input_item.raw_gt_flowed_source_pc_mask & input_item.raw_source_pc_mask) == input_item.raw_gt_flowed_source_pc_mask).all(), f"{input_item.raw_gt_flowed_source_pc_mask} & {input_item.raw_source_pc_mask} != {input_item.raw_gt_flowed_source_pc_mask}"

            # Start with raw source pc, which includes not ground points
            raw_gt_evaluation_mask = input_item.raw_source_pc_mask.clone()
            # Add in the mask for the valid points from the voxelizer
            raw_gt_evaluation_mask[raw_gt_evaluation_mask.clone()] = output_item.pc0_valid_point_mask.to(raw_gt_evaluation_mask.device)
            # Add in the mask for the valid points from the gt flow.
            raw_gt_evaluation_mask = raw_gt_evaluation_mask & input_item.raw_gt_flowed_source_pc_mask

            ############################################################
            # Mask for extracting the eval points for the output flow
            ############################################################

            # Disable the mask for the points that are not in the voxelizer
            output_evaluation_mask = input_item.raw_gt_flowed_source_pc_mask[input_item.raw_source_pc_mask] & output_item.pc0_valid_point_mask.to(raw_gt_evaluation_mask.device)




            source_pc = input_item.raw_source_pc[raw_gt_evaluation_mask]
            gt_flowed_pc = input_item.raw_gt_flowed_source_pc[raw_gt_evaluation_mask]
            gt_flow = gt_flowed_pc - source_pc

            est_flow = output_item.flow[output_evaluation_mask]

            assert gt_flow.shape == est_flow.shape, f"Flow shapes are different: gt {gt_flow.shape} != est {est_flow.shape}"

            loss_difference = est_flow - gt_flow
            diff_l2 = torch.norm(loss_difference, dim=1, p=2).mean()
            total_loss += diff_l2
        return {
            "loss": total_loss,
        }

    def _visualize_loss(self, pc, gt_flow, est_flow):
        import numpy as np
        import open3d as o3d

        pc = pc.detach().cpu().numpy()
        gt_flow = gt_flow.detach().cpu().numpy()
        est_flow = est_flow.detach().cpu().numpy()
        # make open3d visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().point_size = 1.5
        vis.get_render_option().background_color = (0, 0, 0)
        vis.get_render_option().show_coordinate_frame = True
        # set up vector
        vis.get_view_control().set_up([0, 0, 1])

        # Add input PC
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pc_color = np.zeros_like(pc)
        pc_color[:, 0] = 1
        pc_color[:, 1] = 1
        pcd.colors = o3d.utility.Vector3dVector(pc_color)
        vis.add_geometry(pcd)

        # Add gt flowed PC
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc + gt_flow)
        pc_color = np.zeros_like(pc)
        pc_color[:, 1] = 1
        pc_color[:, 2] = 1
        pcd.colors = o3d.utility.Vector3dVector(pc_color)
        vis.add_geometry(pcd)

        # Add line set between pc0 and regressed pc1
        line_set = o3d.geometry.LineSet()
        assert len(pc) == len(est_flow), f"{len(pc)} != {len(est_flow)}"
        line_set_points = np.concatenate([pc, pc + est_flow], axis=0)

        lines = np.array([[i, i + len(est_flow)] for i in range(len(pc))])
        line_set.points = o3d.utility.Vector3dVector(line_set_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(lines))])
        vis.add_geometry(line_set)

        vis.run()


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
        mask = torch.zeros((points.shape[0], ), dtype=torch.bool)
        assert mask_indices.max() < len(mask), f"Mask indices go outside of the tensor range. {mask_indices.max()} >= {len(mask)}"
        mask[mask_indices] = True
        return mask

    def _model_forward(
        self, pc0s: List[torch.FloatTensor], pc1s: List[torch.FloatTensor]
    ) -> List[BucketedSceneFlowOutputItem]:
        """
        Args:
            pc0s: A list (len=batch size) of point source point clouds.
            pc1s: A list (len=batch size) of point target point clouds.
        Returns:
            A list of BucketedSceneFlowOutputItem dataclasses (of length batch size).
        """
        pc0_before_pseudoimages, pc0_voxel_infos_lst = self.embedder(pc0s)
        pc1_before_pseudoimages, pc1_voxel_infos_lst = self.embedder(pc1s)

        grid_flow_pseudoimage = self.backbone(pc0_before_pseudoimages, pc1_before_pseudoimages)
        valid_flows = self.head(
            torch.cat((pc0_before_pseudoimages, pc1_before_pseudoimages), dim=1),
            grid_flow_pseudoimage,
            pc0_voxel_infos_lst,
        )

        batch_output = []

        for raw_pc0, raw_pc1, pc0_voxel_info, pc1_voxel_info, valid_flow in zip(
            pc0s, pc1s, pc0_voxel_infos_lst, pc1_voxel_infos_lst, valid_flows
        ):
            pc0_valid_point_indexes = pc0_voxel_info["point_idxes"]
            pc1_valid_point_indexes = pc1_voxel_info["point_idxes"]

            raw_pc0_valid_point_mask = self._indices_to_mask(raw_pc0, pc0_valid_point_indexes)
            raw_pc1_valid_point_mask = self._indices_to_mask(raw_pc1, pc1_valid_point_indexes)

            raw_flow = torch.zeros_like(raw_pc0)
            raw_flow[raw_pc0_valid_point_mask] = valid_flow

            batch_output.append(
                BucketedSceneFlowOutputItem(
                    flow=raw_flow,  # type: ignore[arg-type]
                    pc0_points=raw_pc0,
                    pc0_valid_point_mask=raw_pc0_valid_point_mask,
                    pc0_warped_points=raw_pc0 + raw_flow,  # type: ignore[arg-type]
                )
            )

        return batch_output

    def forward(
        self, batched_sequence: List[BucketedSceneFlowItem]
    ) -> List[BucketedSceneFlowOutputItem]:
        """
        Args:
            batched_sequence: A list (len=batch size) of BucketedSceneFlowItems.

        Returns:
            A list (len=batch size) of BucketedSceneFlowOutputItem.
        """
        pc0s = [e.source_pc for e in batched_sequence]
        pc1s = [e.target_pc for e in batched_sequence]
        return self._model_forward(pc0s, pc1s)
