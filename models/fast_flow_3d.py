import torch
import torch.nn as nn

import numpy as np
from models.embedders import HardEmbedder, DynamicEmbedder
from models.backbones import FastFlowUNet, FastFlowUNetXL
from models.heads import FastFlowDecoder, FastFlowDecoderStepDown
from pointclouds import from_fixed_array
from pointclouds.losses import warped_pc_loss
from dataloaders import BucketedSceneFlowItem
from typing import Dict, Any, Optional, List
from collections import defaultdict
import time


class FastFlow3DSelfSupervisedLoss():

    def __init__(self, device: str = None):
        super().__init__()

    def _warped_loss(self, model_res):
        estimated_flows = model_res["flow"]
        pc0_points_lst = model_res["pc0_points_lst"]
        pc1_points_lst = model_res["pc1_points_lst"]

        warped_loss = 0
        for flow, pc0_points, pc1_points in zip(estimated_flows,
                                                pc0_points_lst,
                                                pc1_points_lst):
            pc0_warped_to_pc1_points = pc0_points + flow
            warped_loss += warped_pc_loss(pc0_warped_to_pc1_points, pc1_points)
        return warped_loss

    def __call__(self, input_batch, model_res_dict: Dict[str, Dict[str, Any]]):
        return {'loss': self._warped_loss(model_res_dict["forward"])}


class FastFlow3DBucketedLoaderLoss():

    def __init__(self, device: str = None, fast_mover_scale: bool = False):
        super().__init__()
        self.fast_mover_scale = fast_mover_scale

    def __call__(self, input_batch: List[BucketedSceneFlowItem],
                 model_res_dict):

        # Unwrap the model results into the relevant variables
        model_res = model_res_dict["forward"]
        estimated_flows_arr = model_res["flow"]
        valid_point_idxes_arr = model_res["pc0_valid_point_idxes"]

        # Input batch length should be the same as the estimated flows
        assert len(input_batch) == len(
            estimated_flows_arr
        ), f"input_batch {len(input_batch)} != estimated_flows {len(estimated_flows_arr)}"

        total_loss = 0
        # Iterate through the batch
        for input_item, estimated_flow, valid_point_idxes in zip(
                input_batch, estimated_flows_arr, valid_point_idxes_arr):
            source_pc = input_item.source_pc
            gt_flowed_pc = input_item.gt_flowed_source_pc
            gt_flow = gt_flowed_pc - source_pc
            valid_gt_flow = gt_flow[valid_point_idxes]

            flow_difference = estimated_flow - valid_gt_flow
            diff_l2 = torch.norm(flow_difference, dim=1, p=2).mean()
            total_loss += diff_l2

            # self._visualize_loss(source_pc[valid_point_idxes], valid_gt_flow, estimated_flow)

        return {
            "loss": total_loss,
        }

    def _visualize_loss(self, pc, gt_flow, est_flow):
        import open3d as o3d
        import numpy as np
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
        line_set.colors = o3d.utility.Vector3dVector(
            [[0, 0, 1] for _ in range(len(lines))])
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

    def __init__(self,
                 VOXEL_SIZE,
                 PSEUDO_IMAGE_DIMS,
                 POINT_CLOUD_RANGE,
                 FEATURE_CHANNELS,
                 SEQUENCE_LENGTH,
                 bottleneck_head=False,
                 xl_backbone=False) -> None:
        super().__init__()
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH
        assert self.SEQUENCE_LENGTH == 2, "This implementation only supports a sequence length of 2."
        self.embedder = DynamicEmbedder(voxel_size=VOXEL_SIZE,
                                        pseudo_image_dims=PSEUDO_IMAGE_DIMS,
                                        point_cloud_range=POINT_CLOUD_RANGE,
                                        feat_channels=FEATURE_CHANNELS)
        if xl_backbone:
            self.backbone = FastFlowUNetXL()
        else:
            self.backbone = FastFlowUNet()
        if bottleneck_head:
            self.head = FastFlowDecoderStepDown(
                voxel_pillar_size=VOXEL_SIZE[:2], num_stepdowns=3)
        else:
            self.head = FastFlowDecoder(pseudoimage_channels=FEATURE_CHANNELS *
                                        2)

    def _model_forward(self, pc0s, pc1s):

        before_forward = time.time()
        pc0_before_pseudoimages, pc0_voxel_infos_lst = self.embedder(pc0s)
        pc1_before_pseudoimages, pc1_voxel_infos_lst = self.embedder(pc1s)

        grid_flow_pseudoimage = self.backbone(pc0_before_pseudoimages,
                                              pc1_before_pseudoimages)
        flows = self.head(
            torch.cat((pc0_before_pseudoimages, pc1_before_pseudoimages),
                      dim=1), grid_flow_pseudoimage, pc0_voxel_infos_lst)
        after_forward = time.time()

        pc0_points_lst = [e["points"] for e in pc0_voxel_infos_lst]
        pc0_valid_point_idxes = [e["point_idxes"] for e in pc0_voxel_infos_lst]
        pc1_points_lst = [e["points"] for e in pc1_voxel_infos_lst]
        pc1_valid_point_idxes = [e["point_idxes"] for e in pc1_voxel_infos_lst]

        pc0_warped_pc1_points_lst = [
            pc0_points + flow
            for pc0_points, flow in zip(pc0_points_lst, flows)
        ]

        return {
            "flow": flows,
            "batch_delta_time": after_forward - before_forward,
            "pc0_points_lst": pc0_points_lst,
            "pc0_warped_pc1_points_lst": pc0_warped_pc1_points_lst,
            "pc0_valid_point_idxes": pc0_valid_point_idxes,
            "pc1_points_lst": pc1_points_lst,
            "pc1_valid_point_idxes": pc1_valid_point_idxes
        }

    def forward(self, batched_sequence: List[BucketedSceneFlowItem]):
        pc0s = [e.source_pc for e in batched_sequence]
        pc1s = [e.target_pc for e in batched_sequence]
        model_res = self._model_forward(pc0s, pc1s)

        ret_dict = {"forward": model_res}

        return ret_dict
