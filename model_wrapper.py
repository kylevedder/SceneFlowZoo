import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import models

import pytorch_lightning as pl
import torchmetrics
from typing import Dict, List, Tuple

CATEGORY_NAMES = {
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

SPEED_BUCKET_SPLITS_METERS_PER_SECOND = [0, 0.1, 1.0, np.inf]


class EndpointDistanceMetricRawTorch():

    def __init__(self, class_id_to_name_map: Dict[int, str],
                 speed_bucket_splits_meters_per_second: List[float]):
        self.class_index_to_name_map = {}
        self.class_id_to_index_map = {}
        for cls_index, (cls_id,
                        cls_name) in enumerate(class_id_to_name_map.items()):
            self.class_index_to_name_map[cls_index] = cls_name
            self.class_id_to_index_map[cls_id] = cls_index

        self.per_class_total_error_sum = torch.zeros(
            (len(class_id_to_name_map)), dtype=torch.float)
        self.per_class_total_error_count = torch.zeros(
            (len(class_id_to_name_map)), dtype=torch.long)

        self.speed_bucket_splits_meters_per_second = speed_bucket_splits_meters_per_second
        speed_bucket_bounds = self.speed_bucket_bounds()
        self.speed_bucket_errors = torch.zeros(len(speed_bucket_bounds),
                                               dtype=torch.float)
        self.speed_bucket_counts = torch.zeros(len(speed_bucket_bounds),
                                               dtype=torch.long)

    def to(self, device):
        self.per_class_total_error_sum = self.per_class_total_error_sum.to(
            device)
        self.per_class_total_error_count = self.per_class_total_error_count.to(
            device)
        self.speed_bucket_errors = self.speed_bucket_errors.to(device)

    def speed_bucket_bounds(self) -> List[Tuple[float, float]]:
        return list(
            zip(self.speed_bucket_splits_meters_per_second,
                self.speed_bucket_splits_meters_per_second[1:]))

    def update_class_error(self, class_id: int, endpoint_error,
                           endpoint_count: int):
        class_index = self.class_id_to_index_map[class_id]
        self.per_class_total_error_sum[class_index] += endpoint_error
        self.per_class_total_error_count[class_index] += endpoint_count

    def update_speed_bucket_error(self, gt_speeds: torch.Tensor,
                                  endpoint_errors: torch.Tensor):
        for bucket_idx, (lower_speed_bound, upper_speed_bound) in enumerate(
                self.speed_bucket_bounds()):
            bucket_mask = (gt_speeds >=
                           lower_speed_bound) & (gt_speeds < upper_speed_bound)
            bucket_errors = endpoint_errors[bucket_mask]
            self.speed_bucket_errors[bucket_idx] += torch.sum(bucket_errors)
            self.speed_bucket_counts[bucket_idx] += len(bucket_errors)

    # def compute(self):
    #     return self.per_class_total_error_sum / self.per_class_total_error_count


class ModelWrapper(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.model = getattr(models, cfg.model.name)(**cfg.model.args)

        if not hasattr(cfg, "is_trainable") or cfg.is_trainable:
            self.loss_fn = getattr(models,
                                   cfg.loss_fn.name)(**cfg.loss_fn.args)

        self.lr = cfg.learning_rate
        if hasattr(cfg, "train_forward_args"):
            self.train_forward_args = cfg.train_forward_args
        else:
            self.train_forward_args = {}

        if hasattr(cfg, "val_forward_args"):
            self.val_forward_args = cfg.val_forward_args
        else:
            self.val_forward_args = {}

        self.has_labels = True if not hasattr(cfg,
                                              "has_labels") else cfg.has_labels

        self.metric = EndpointDistanceMetricRawTorch(
            CATEGORY_NAMES, SPEED_BUCKET_SPLITS_METERS_PER_SECOND)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, input_batch, batch_idx):
        model_res = self.model(input_batch, **self.train_forward_args)
        loss_res = self.loss_fn(input_batch, model_res)
        loss = loss_res.pop("loss")
        self.log("train/loss", loss, on_step=True)
        for k, v in loss_res.items():
            self.log(f"train/{k}", v, on_step=True)
        return {"loss": loss}

    def _visualize_regressed_ground_truth_pcs(self, pc0_pc, pc1_pc,
                                              regressed_flowed_pc0_to_pc1,
                                              ground_truth_flowed_pc0_to_pc1):
        import open3d as o3d
        import numpy as np
        pc0_pc = pc0_pc.cpu().numpy()
        pc1_pc = pc1_pc.cpu().numpy()
        regressed_flowed_pc0_to_pc1 = regressed_flowed_pc0_to_pc1.cpu().numpy()
        ground_truth_flowed_pc0_to_pc1 = ground_truth_flowed_pc0_to_pc1.cpu(
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

        # Add line set between pc0 and gt pc1
        line_set = o3d.geometry.LineSet()
        assert len(pc0_pc) == len(
            ground_truth_flowed_pc0_to_pc1
        ), f"{len(pc0_pc)} != {len(ground_truth_flowed_pc0_to_pc1)}"
        line_set_points = np.concatenate(
            [pc0_pc, ground_truth_flowed_pc0_to_pc1], axis=0)

        lines = np.array([[i, i + len(ground_truth_flowed_pc0_to_pc1)]
                          for i in range(len(pc0_pc))])
        line_set.points = o3d.utility.Vector3dVector(line_set_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(
            [[0, 1, 0] for _ in range(len(lines))])
        vis.add_geometry(line_set)

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

    def validation_step(self, input_batch, batch_idx):
        output_batch = self.model(input_batch, **self.val_forward_args)["forward"]
        self.metric.to(self.device)

        if not self.has_labels:
            return

        # Decode the mini-batch.
        for minibatch_idx, (pc_array, flowed_pc_array, regressed_flow,
                            pc0_valid_point_idxes, pc1_valid_point_idxes,
                            class_info) in enumerate(
                                zip(input_batch["pc_array_stack"],
                                    input_batch["flowed_pc_array_stack"],
                                    output_batch["flow"],
                                    output_batch["pc0_valid_point_idxes"],
                                    output_batch["pc1_valid_point_idxes"],
                                    input_batch["pc_class_mask_stack"])):
            # This is written to support an arbitrary sequence length, but we only want to compute a flow
            # off of the last frame.
            pc0_pc = pc_array[-2][pc0_valid_point_idxes]
            pc1_pc = pc_array[-1][pc1_valid_point_idxes]
            ground_truth_flowed_pc0_to_pc1 = flowed_pc_array[-2][
                pc0_valid_point_idxes]
            pc0_pc_class_info = class_info[-2][pc0_valid_point_idxes]

            assert pc0_pc.shape == ground_truth_flowed_pc0_to_pc1.shape, f"The input and ground truth pointclouds are not the same shape. {pc0_pc.shape} != {ground_truth_flowed_pc0_to_pc1.shape}"
            assert pc0_pc.shape == regressed_flow.shape, f"The input pc and output flow are not the same shape. {pc0_pc.shape} != {regressed_flow.shape}"

            regressed_flowed_pc0_to_pc1 = pc0_pc + regressed_flow

            assert regressed_flowed_pc0_to_pc1.shape == ground_truth_flowed_pc0_to_pc1.shape, f"The regressed and ground truth flowed pointclouds are not the same shape. {regressed_flowed_pc0_to_pc1.shape} != {ground_truth_flowed_pc0_to_pc1.shape}"

            # print("Batch", batch_idx, ":", torch.mean(torch.norm(regressed_flow, dim=1)).item())

            # if batch_idx % 64 == 0 and minibatch_idx == 0:
            #     self._visualize_regressed_ground_truth_pcs(
            #         pc0_pc, pc1_pc, regressed_flowed_pc0_to_pc1,
            #         ground_truth_flowed_pc0_to_pc1)

            # Per point L2 error between the regressed flowed pointcloud and the ground truth flowed pointcloud.
            endpoint_errors = torch.norm(regressed_flowed_pc0_to_pc1 -
                                         ground_truth_flowed_pc0_to_pc1,
                                         dim=1,
                                         p=2)

            # ======================== Compute Per-Class Metrics ========================

            for cls_id in torch.unique(pc0_pc_class_info):
                cls_mask = (pc0_pc_class_info == cls_id)
                cls_distance_array = endpoint_errors[cls_mask]
                total_distance = torch.sum(cls_distance_array)
                self.metric.update_class_error(cls_id.item(), total_distance,
                                               cls_distance_array.shape[0])

            # ======================== Compute Speed Bucket Metrics ========================

            # Scale the ground truth flow, computed over 1/10th of a second, to be over 1 second.
            ground_truth_speeds = torch.norm(
                (ground_truth_flowed_pc0_to_pc1 - pc0_pc) * 10.0, dim=1, p=2)
            self.metric.update_speed_bucket_error(ground_truth_speeds,
                                                  endpoint_errors)

    def validation_epoch_end(self, batch_parts):
        import time
        before_gather = time.time()
        world_per_class_total_error_sum = self.all_gather(
            self.metric.per_class_total_error_sum)
        world_per_class_total_error_count = self.all_gather(
            self.metric.per_class_total_error_count)
        world_speed_bucket_errors = self.all_gather(
            self.metric.speed_bucket_errors)
        world_speed_bucket_counts = self.all_gather(
            self.metric.speed_bucket_counts)

        after_gather = time.time()

        print(
            f"Rank {self.global_rank} gathers done in {after_gather - before_gather}."
        )

        if self.global_rank != 0:
            return {}

        # Sum the per-GPU metrics across the world axis.
        per_class_total_error_sum = torch.sum(world_per_class_total_error_sum,
                                              dim=0)
        per_class_total_error_count = torch.sum(
            world_per_class_total_error_count, dim=0)

        speed_bucket_errors = torch.sum(world_speed_bucket_errors, dim=0)
        speed_bucket_counts = torch.sum(world_speed_bucket_counts, dim=0)

        # ======================== Log Per-Class Metrics ========================

        cls_avg_errors = per_class_total_error_sum / per_class_total_error_count

        perf_dict = {}
        for cls_idx, cls_avg_error in enumerate(cls_avg_errors):
            category_name = self.metric.class_index_to_name_map[cls_idx]
            perf_dict[category_name] = cls_avg_error.item()
            self.log(f"val/{category_name}",
                     cls_avg_error,
                     sync_dist=False,
                     rank_zero_only=True)

        bucket_avg_errors = speed_bucket_errors / speed_bucket_counts

        # ======================== Log Speed Bucket Metrics ========================
        for bucket_avg_error, (lower_bound, upper_bound) in zip(
                bucket_avg_errors,
                self.metric.speed_bucket_bounds()):
            category_name = f"{lower_bound}-{upper_bound}_speed_bucket"
            perf_dict[category_name] = bucket_avg_error.item()
            self.log(f"val/{category_name}",
                     bucket_avg_error,
                     sync_dist=False,
                     rank_zero_only=True)

        return perf_dict