import torch
import torch.nn as nn

import numpy as np
from models.embedders import HardEmbedder, DynamicEmbedder
from models.backbones import FastFlowUNet
from models.heads import FastFlowDecoder
from pointclouds import PointCloud, warped_pc_loss, pc0_to_pc1_distance

from typing import Dict, Any, Optional
from collections import defaultdict

CATEGORY_MAP = {
    -1: 'BACKGROUND',
    0: 'ANIMAL',
    1: 'ARTICULATED_BUS',
    2: 'BICYCLE',
    3: 'BICYCLIST',
    4: 'BOLLARD',
    5: 'BOX_TRUCK',
    6: 'BUS',
    7: 'CONSTRUCTION_BARREL',
    8: 'CONSTRUCTION_CONE',
    9: 'DOG',
    10: 'LARGE_VEHICLE',
    11: 'MESSAGE_BOARD_TRAILER',
    12: 'MOBILE_PEDESTRIAN_CROSSING_SIGN',
    13: 'MOTORCYCLE',
    14: 'MOTORCYCLIST',
    15: 'OFFICIAL_SIGNALER',
    16: 'PEDESTRIAN',
    17: 'RAILED_VEHICLE',
    18: 'REGULAR_VEHICLE',
    19: 'SCHOOL_BUS',
    20: 'SIGN',
    21: 'STOP_SIGN',
    22: 'STROLLER',
    23: 'TRAFFIC_LIGHT_TRAILER',
    24: 'TRUCK',
    25: 'TRUCK_CAB',
    26: 'VEHICULAR_TRAILER',
    27: 'WHEELCHAIR',
    28: 'WHEELED_DEVICE',
    29: 'WHEELED_RIDER'
}


class FastFlow3DLoss():

    def __init__(self, device: str = None):
        super().__init__()

    def __call__(self, model_res):
        total_loss = 0
        flows = model_res["flow"]
        pc0_points_lst = model_res["pc0_points_lst"]
        pc1_points_lst = model_res["pc1_points_lst"]
        for flow, pc0_points, pc1_points in zip(flows, pc0_points_lst,
                                                pc1_points_lst):
            warped_pc1_points = pc0_points + flow

            loss = warped_pc_loss(pc1_points, warped_pc1_points)
            total_loss += loss
        return total_loss


class FastFlow3DTestLoss():

    def __init__(self, device: str = None):
        super().__init__()
        self.input_lst = []
        self.output_lst = []

    def _data_entries_to_cpu(self, data):
        if isinstance(data, torch.Tensor):
            data = data.cpu()
            return data
        elif isinstance(data, list) or isinstance(data, tuple):
            return [self._data_entries_to_cpu(d) for d in data]
        elif isinstance(data, dict):
            return {k: self._data_entries_to_cpu(v) for k, v in data.items()}
        else:
            return data

    def accumulate(self, batch_idx: int, input_batch, output_batch):
        cpu_input_batch = self._data_entries_to_cpu(input_batch)
        self.input_lst.append(cpu_input_batch)
        cpu_output_batch = self._data_entries_to_cpu(output_batch)
        self.output_lst.append(cpu_output_batch)

    def _compute_per_class_average_error(self, visualize: bool = False):
        # INPUTS: pointclouds and the ground truth flowed pointclouds. These pointclouds are in a fixed size
        # representation, meaning they have NaNs. To decode these into pointclouds that _also_ correspond to
        # the detection area of the pointclouds, we need to use use the valid index mask provided by the output.
        #
        # OUTPUTS: The output describes the predicted flow (offset from the input pointcloud) which needs
        # to be applied to the input pointclouds and then compared against the ground truth. It also describes
        # valid index mask to be applied to the input pointclouds.
        per_class_average_frame_error = defaultdict(list)
        per_class_total_error_sum = defaultdict(float)
        per_class_total_error_count = defaultdict(int)

        # The input and output batches are lists of mini-batches. Decode the full batch.
        for input_batch, output_batch in zip(self.input_lst, self.output_lst):
            assert len(input_batch["pc_array_stack"]) == len(
                output_batch["flow"]
            ), f"The input and output batches are not the same length. {len(input_batch['pc_array_stack'])} != {len(output_batch['flow'])}"

            # Decode the mini-batch.
            for pc_array, flowed_pc_array, regressed_flow, pc0_valid_point_idxes, class_info in zip(
                    input_batch["pc_array_stack"],
                    input_batch["flowed_pc_array_stack"], output_batch["flow"],
                    output_batch["pc0_valid_point_idxes"],
                    input_batch["pc_class_mask_stack"]):
                # This is written to support an arbitrary sequence length, but we only want to compute a flow
                # off of the last frame.
                pc0_pc = pc_array[-2][pc0_valid_point_idxes]
                ground_truth_flowed_pc0_to_pc1 = flowed_pc_array[-2][
                    pc0_valid_point_idxes]
                pc0_pc_class_info = class_info[-2][pc0_valid_point_idxes]

                assert pc0_pc.shape == ground_truth_flowed_pc0_to_pc1.shape, f"The input and ground truth pointclouds are not the same shape. {pc0_pc.shape} != {ground_truth_flowed_pc0_to_pc1.shape}"
                assert pc0_pc.shape == regressed_flow.shape, f"The input pc and output flow are not the same shape. {pc0_pc.shape} != {regressed_flow.shape}"

                regressed_flowed_pc0_to_pc1 = pc0_pc + regressed_flow

                assert regressed_flowed_pc0_to_pc1.shape == ground_truth_flowed_pc0_to_pc1.shape, f"The regressed and ground truth flowed pointclouds are not the same shape. {regressed_flowed_pc0_to_pc1.shape} != {ground_truth_flowed_pc0_to_pc1.shape}"

                # Per point L2 error between the regressed flowed pointcloud and the ground truth flowed pointcloud.
                distance_array = torch.norm(regressed_flowed_pc0_to_pc1 -
                                            ground_truth_flowed_pc0_to_pc1,
                                            dim=1,
                                            p=2)

                if visualize:
                    self._visualize_regressed_ground_truth_pcs(
                        pc0_pc, regressed_flowed_pc0_to_pc1,
                        ground_truth_flowed_pc0_to_pc1)

                # Compute average error per class.
                for cls_id in torch.unique(pc0_pc_class_info):
                    cls_mask = pc0_pc_class_info == cls_id
                    cls_distance_array = distance_array[cls_mask]
                    cls_avg_distance = torch.mean(cls_distance_array)
                    cls_id = int(cls_id.item())
                    per_class_average_frame_error[cls_id].append(
                        cls_avg_distance)
                    per_class_total_error_sum[cls_id] += torch.sum(
                        cls_distance_array)
                    per_class_total_error_count[
                        cls_id] += cls_distance_array.shape[0]

        return per_class_average_frame_error, per_class_total_error_sum, per_class_total_error_count

    def _visualize_regressed_ground_truth_pcs(self, pc0_pc,
                                              regressed_flowed_pc0_to_pc1,
                                              ground_truth_flowed_pc0_to_pc1):
        import open3d as o3d
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
        pc_color[:, 0] = 1.0
        pcd.colors = o3d.utility.Vector3dVector(pc_color)
        vis.add_geometry(pcd)

        # Add gt PC
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(
            ground_truth_flowed_pc0_to_pc1)
        gt_pc_color = np.zeros_like(ground_truth_flowed_pc0_to_pc1)
        gt_pc_color[:, 1] = 1.0
        gt_pcd.colors = o3d.utility.Vector3dVector(gt_pc_color)
        vis.add_geometry(gt_pcd)

        # Add regressed PC
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(regressed_flowed_pc0_to_pc1)
        pc_color = np.zeros_like(regressed_flowed_pc0_to_pc1)
        pc_color[:, 2] = 1.0
        pcd.colors = o3d.utility.Vector3dVector(pc_color)
        vis.add_geometry(pcd)

        vis.run()

    def finalize(self,
                 epoch: Optional[int] = None,
                 writer: Optional['SummaryWriter'] = None):
        print("Finalizing evaluation...")
        print("Length of input list: ", len(self.input_lst))
        print("Length of output list: ", len(self.output_lst))
        per_class_average_error, per_class_total_error_sum, per_class_total_error_count = self._compute_per_class_average_error(
        )

        print("Average frame error per class:")
        for cls_id, cls_error in sorted(per_class_average_error.items()):
            class_name = CATEGORY_MAP[cls_id]
            error = torch.mean(torch.stack(cls_error)).item()
            print(f"Class {class_name} average error: {error}")
            if epoch is not None and writer is not None:
                writer.add_scalar(f"test/frame_average_error/{class_name}",
                                  error, epoch)

        print("Overall error per class:")
        for cls_id, cls_error in sorted(per_class_total_error_sum.items()):
            class_name = CATEGORY_MAP[cls_id]
            error_sum = per_class_total_error_sum[cls_id]
            error_count = per_class_total_error_count[cls_id]
            error = error_sum / error_count
            print(f"Class {class_name} average error: {error}")
            if epoch is not None and writer is not None:
                writer.add_scalar(f"test/total_average_error/{class_name}",
                                  error, epoch)


class FastFlow3D(nn.Module):
    """
    FastFlow3D based on the paper:
    https://arxiv.org/abs/2103.01306v5

    Note that there are several small differences between this implementation and the paper:
     - We use a different loss function (predict flow for P_-1 to P_0 instead of P_0 to and 
       unseen P_1); referred to as pc0 and pc1 in the code.
    """

    def __init__(self, VOXEL_SIZE, PSEUDO_IMAGE_DIMS,
                 POINT_CLOUD_RANGE, MAX_POINTS_PER_VOXEL, FEATURE_CHANNELS,
                 SEQUENCE_LENGTH) -> None:
        super().__init__()
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

        pc0_points_lst = [
            points for points, _, _ in pc0_points_coordinates_lst
        ]
        pc0_valid_point_idxes = [
            valid_point_idxes
            for _, _, valid_point_idxes in pc0_points_coordinates_lst
        ]
        pc1_points_lst = [
            points for points, _, _ in pc1_points_coordinates_lst
        ]
        pc1_valid_point_idxes = [
            valid_point_idxes
            for _, _, valid_point_idxes in pc1_points_coordinates_lst
        ]

        return {
            "flow": flows,
            "pc0_points_lst": pc0_points_lst,
            "pc0_valid_point_idxes": pc0_valid_point_idxes,
            "pc1_points_lst": pc1_points_lst,
            "pc1_valid_point_idxes": pc1_valid_point_idxes
        }
