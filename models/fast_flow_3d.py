import torch
import torch.nn as nn

import numpy as np
from models.embedders import HardEmbedder, DynamicEmbedder
from models.backbones import FastFlowUNet, FastFlowUNetXL
from models.heads import FastFlowDecoder, FastFlowDecoderStepDown
from pointclouds import from_fixed_array
from pointclouds.losses import warped_pc_loss

from typing import Dict, Any, Optional
from collections import defaultdict
import time


class FastFlow3DSelfSupervisedLoss():

    def __init__(self, warp_upscale: float, device: str = None):
        super().__init__()
        self.warp_upscale = warp_upscale

    def _warped_loss(self, model_res):
        flows = model_res["flow"]
        pc0_points_lst = model_res["pc0_points_lst"]
        pc1_points_lst = model_res["pc1_points_lst"]

        warped_loss = 0
        for flow, pc0_points, pc1_points in zip(flows, pc0_points_lst,
                                                pc1_points_lst):
            pc0_warped_to_pc1_points = pc0_points + flow
            warped_loss += warped_pc_loss(pc0_warped_to_pc1_points,
                                          pc1_points) * self.warp_upscale
        return warped_loss

    def _triangle_loss(self, forward_res, backward_res):
        forward_flows = forward_res["flow"]
        backward_flows = backward_res["flow"]
        # These are the idxes of the forward flow present in the backwards flow
        valid_forward_flow_idxes = backward_res["pc0_valid_point_idxes"]

        triangle_loss = 0
        for forward_flow, backward_flow, valid_forward_flow_idx in zip(
                forward_flows, backward_flows, valid_forward_flow_idxes):
            forward_flow_valid_in_backward_flow = forward_flow[
                valid_forward_flow_idx]

            triangle_error = torch.norm(forward_flow_valid_in_backward_flow +
                                        backward_flow,
                                        dim=1,
                                        p=2)
            triangle_loss += torch.mean(triangle_error)
        return triangle_loss

    def _symmetry_loss(self, model_res, flipped_res, idx):
        flows = model_res["flow"]
        flipped_flows = flipped_res["flow"]
        symmetry_loss = 0
        for flow, flipped_flow in zip(flows, flipped_flows):
            # Other than the y axis, the flows should be the same
            # Negate the y axis of the flipped flow, then subtract the two flows should produce a zero flow
            flipped_flow[:, idx] = -flipped_flow[:, idx]
            symmetry_loss += torch.mean(
                torch.norm(flow - flipped_flow, dim=1, p=2))

        return symmetry_loss

    def __call__(self, input_batch, model_res_dict: Dict[str, Dict[str, Any]]):
        warped_loss = self._warped_loss(model_res_dict["forward"])
        # self._visualize_regressed_ground_truth_pcs(model_res_dict["forward"])

        res_dict = {
            "warped_loss": warped_loss,
        }

        total_loss = warped_loss

        if "backward" in model_res_dict:
            triangle_loss = self._triangle_loss(model_res_dict["forward"],
                                                model_res_dict["backward"])
            total_loss += triangle_loss
            res_dict["triangle_loss"] = triangle_loss

        if "symmetry_x" in model_res_dict:
            symmetry_loss = self._symmetry_loss(model_res_dict["forward"],
                                                model_res_dict["symmetry_x"],
                                                0)
            total_loss += symmetry_loss
            res_dict["symmetry_x_loss"] = symmetry_loss

        if "symmetry_y" in model_res_dict:
            symmetry_loss = self._symmetry_loss(model_res_dict["forward"],
                                                model_res_dict["symmetry_y"],
                                                1)
            total_loss += symmetry_loss
            res_dict["symmetry_y_loss"] = symmetry_loss

        res_dict["loss"] = total_loss

        return res_dict

    def _visualize_regressed_ground_truth_pcs(self, model_res):
        regressed_flowed_pc0_to_pc1 = model_res["flow"]
        pc0_pc = model_res["pc0_points_lst"]
        pc1_pc = model_res["pc1_points_lst"]
        import open3d as o3d
        import numpy as np
        pc0_pc = pc0_pc.detach().cpu().numpy()
        pc1_pc = pc1_pc.detach().cpu().numpy()
        regressed_flowed_pc0_to_pc1 = regressed_flowed_pc0_to_pc1.detach().cpu(
        ).numpy()
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
        pcd.points = o3d.utility.Vector3dVector(pc0_pc)
        pc_color = np.zeros_like(pc0_pc)
        pc_color[:, 0] = 1
        pc_color[:, 1] = 1
        pcd.colors = o3d.utility.Vector3dVector(pc_color)
        vis.add_geometry(pcd)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc1_pc)
        pc_color = np.zeros_like(pc1_pc)
        pc_color[:, 1] = 1
        pc_color[:, 2] = 1
        pcd.colors = o3d.utility.Vector3dVector(pc_color)
        vis.add_geometry(pcd)

        # Add line set between pc0 and regressed pc1
        line_set = o3d.geometry.LineSet()
        assert len(pc0_pc) == len(
            regressed_flowed_pc0_to_pc1
        ), f"{len(pc0_pc)} != {len(regressed_flowed_pc0_to_pc1)}"
        line_set_points = np.concatenate([pc0_pc, regressed_flowed_pc0_to_pc1],
                                         axis=0)

        lines = np.array([[i, i + len(regressed_flowed_pc0_to_pc1)]
                          for i in range(len(pc0_pc))])
        line_set.points = o3d.utility.Vector3dVector(line_set_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(
            [[0, 0, 1] for _ in range(len(lines))])
        vis.add_geometry(line_set)

        vis.run()


class FastFlow3DDistillationLoss():

    def __init__(self, device: str = None, fast_mover_scale: bool = False):
        super().__init__()
        self.fast_mover_scale = fast_mover_scale

    def __call__(self, input_batch, model_res_dict):
        model_res = model_res_dict["forward"]
        estimated_flows = model_res["flow"]

        gt_flow_array_stack = input_batch['flow_array_stack']
        # self._visualize_regressed_ground_truth_pcs(model_res,
        #                                            gt_flow_array_stack)

        assert len(estimated_flows) <= len(
            gt_flow_array_stack
        ), f"estimated_flows {len(estimated_flows)} > gt_flow_array_stack {len(gt_flow_array_stack)}"

        total_loss = 0
        # Iterate through the batch
        for est_flow, gt_flow_array in zip(estimated_flows,
                                           gt_flow_array_stack):
            gt_flow = from_fixed_array(gt_flow_array[0])
            assert est_flow.shape == gt_flow.shape, f"estimated_flow {est_flow.shape} != ground_truth_flow {gt_flow.shape}"

            importance_scale = torch.ones_like(gt_flow[:, 0])
            if self.fast_mover_scale:
                # Compute the importance scale using m/s speed.
                gt_speed = torch.norm(gt_flow, dim=1, p=2) * 10.0
                mins = torch.ones_like(gt_speed) * 0.1
                maxs = torch.ones_like(gt_speed)
                # Plot \max\left(0.1,\min\left(1,1.8x-0.8\right)\right) in Desmos
                importance_scale = torch.min(
                    mins, torch.max(1.8 * gt_speed - 0.8, maxs))

            total_loss += (torch.norm(est_flow - gt_flow, dim=1, p=2) *
                           importance_scale).mean()

        return {
            "loss": total_loss,
        }

    def _visualize_regressed_ground_truth_pcs(self, model_res,
                                              gt_flow_array_stack):
        regressed_flowed_pc0_to_pc1_lst = model_res["flow"]
        pc0_pc_lst = model_res["pc0_points_lst"]
        pc1_pc_lst = model_res["pc1_points_lst"]
        import open3d as o3d
        import numpy as np
        for regressed_flowed_pc0_to_pc1, pc0_pc, pc1_pc, gt_flow_array in zip(
                regressed_flowed_pc0_to_pc1_lst, pc0_pc_lst, pc1_pc_lst,
                gt_flow_array_stack):
            gt_flow = from_fixed_array(gt_flow_array[0])
            pc0_pc = pc0_pc.detach().cpu().numpy()
            pc1_pc = pc1_pc.detach().cpu().numpy()
            regressed_flowed_pc0_to_pc1 = pc0_pc + gt_flow.detach().cpu(
            ).numpy()
            #regressed_flowed_pc0_to_pc1.detach().cpu().numpy()
            # breakpoint()

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
            pcd.points = o3d.utility.Vector3dVector(pc0_pc)
            pc_color = np.zeros_like(pc0_pc)
            pc_color[:, 0] = 1
            pc_color[:, 1] = 1
            pcd.colors = o3d.utility.Vector3dVector(pc_color)
            vis.add_geometry(pcd)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc1_pc)
            pc_color = np.zeros_like(pc1_pc)
            pc_color[:, 1] = 1
            pc_color[:, 2] = 1
            pcd.colors = o3d.utility.Vector3dVector(pc_color)
            vis.add_geometry(pcd)

            # Add line set between pc0 and regressed pc1
            line_set = o3d.geometry.LineSet()
            assert len(pc0_pc) == len(
                regressed_flowed_pc0_to_pc1
            ), f"{len(pc0_pc)} != {len(regressed_flowed_pc0_to_pc1)}"
            line_set_points = np.concatenate(
                [pc0_pc, regressed_flowed_pc0_to_pc1], axis=0)

            lines = np.array([[i, i + len(regressed_flowed_pc0_to_pc1)]
                              for i in range(len(pc0_pc))])
            line_set.points = o3d.utility.Vector3dVector(line_set_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(
                [[0, 0, 1] for _ in range(len(lines))])
            vis.add_geometry(line_set)

            vis.run()


class FastFlow3DSupervisedLoss():

    def __init__(self, device: str = None, scale_background: bool = True, scale_speed : bool = False):
        super().__init__()
        self._scale_background = scale_background
        self._scale_speed = scale_speed

        assert not (self._scale_background and self._scale_speed), "Cannot scale both background and speed"

    def __call__(self, input_batch, model_res_dict):
        model_res = model_res_dict["forward"]
        estimated_flows = model_res["flow"]

        pc0_valid_point_idxes = model_res["pc0_valid_point_idxes"]
        input_pcs = input_batch['pc_array_stack']
        flowed_input_pcs = input_batch['flowed_pc_array_stack']
        pc_classes = input_batch['pc_class_mask_stack']
        input_flows = flowed_input_pcs - input_pcs

        total_loss = 0
        # Iterate through the batch
        for estimated_flow, input_flow, pc0_valid_point_idx, pc_class_arr in zip(
                estimated_flows, input_flows, pc0_valid_point_idxes,
                pc_classes):
            ground_truth_flow = input_flow[0, pc0_valid_point_idx]

            error = torch.norm(estimated_flow - ground_truth_flow, dim=1, p=2)

            if self._scale_background:
                classes = pc_class_arr[0, pc0_valid_point_idx]
                # Background class is -1
                is_foreground_class = (classes >= 0)
                # We want to build a scalar array where the background points are 0.1 and the foreground points are 1
                # This is because we want to downweight the background points by 10x
                # We do this simply by converting the bool, where the foreground points are 1, to float, and then
                # multiplying by 0.9 and adding 0.1 to get a scalar in {0.1, 1}
                background_scalar = is_foreground_class.float() * 0.9 + 0.1
                error = error * background_scalar

            if self._scale_speed:
                # Compute the importance scale using m/s speed.
                gt_speed = torch.norm(ground_truth_flow, dim=1, p=2) * 10.0
                mins = torch.ones_like(gt_speed) * 0.1
                maxs = torch.ones_like(gt_speed)
                # Plot \max\left(0.1,\min\left(1,1.8x-0.8\right)\right) in Desmos
                importance_scale = torch.min(
                    mins, torch.max(1.8 * gt_speed - 0.8, maxs))
                error = error * importance_scale

            total_loss += error.mean()

        return {"loss": total_loss}


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

    def forward(self,
                batched_sequence: Dict[str, torch.Tensor],
                compute_cycle=False,
                compute_symmetry_x=False,
                compute_symmetry_y=False):
        pc_arrays = batched_sequence['pc_array_stack']
        pc0s = pc_arrays[:, 0]
        pc1s = pc_arrays[:, 1]
        model_res = self._model_forward(pc0s, pc1s)

        ret_dict = {"forward": model_res}

        if compute_cycle:
            # The warped pointcloud, original pointcloud should be the input to the model
            pc0_warped_pc1_points_lst = model_res["pc0_warped_pc1_points_lst"]
            pc0_points_lst = model_res["pc0_points_lst"]
            # Some of the warped points may be outside the pseudoimage range, causing them to be clipped.
            # When we compute this reverse flow, we need to solve for the original points that were warped to the clipped points.
            backward_model_res = self._model_forward(pc0_warped_pc1_points_lst,
                                                     pc0_points_lst)
            # model_res["reverse_flow"] = backward_model_res["flow"]
            # model_res[
            #     "flow_valid_point_idxes_for_reverse_flow"] = backward_model_res[
            #         "pc0_valid_point_idxes"]
            ret_dict["backward"] = backward_model_res

        if compute_symmetry_x:
            pc0s_sym = pc0s.clone()
            pc0s_sym[:, :, 0] *= -1
            pc1s_sym = pc1s.clone()
            pc1s_sym[:, :, 0] *= -1
            model_res_sym = self._model_forward(pc0s_sym, pc1s_sym)
            ret_dict["symmetry_x"] = model_res_sym

        if compute_symmetry_y:
            pc0s_sym = pc0s.clone()
            pc0s_sym[:, :, 1] *= -1
            pc1s_sym = pc1s.clone()
            pc1s_sym[:, :, 1] *= -1
            model_res_sym = self._model_forward(pc0s_sym, pc1s_sym)
            ret_dict["symmetry_y"] = model_res_sym

        return ret_dict
