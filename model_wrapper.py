import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import models

import pytorch_lightning as pl
import torchmetrics
from typing import Dict, List, Tuple
from loader_utils import save_pickle

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
ENDPOINT_ERROR_SPLITS_METERS = [0, 0.05, 0.1, np.inf]


class EndpointDistanceMetricRawTorch():

    def __init__(self,
                 class_id_to_name_map: Dict[int, str],
                 speed_bucket_splits_meters_per_second: List[float],
                 endpoint_error_splits_meters: List[float],
                 per_frame_to_per_second_scale_factor: float = 10.0):
        self.class_index_to_name_map = {}
        self.class_id_to_index_map = {}
        for cls_index, (cls_id,
                        cls_name) in enumerate(class_id_to_name_map.items()):
            self.class_index_to_name_map[cls_index] = cls_name
            self.class_id_to_index_map[cls_id] = cls_index

        self.speed_bucket_splits_meters_per_second = speed_bucket_splits_meters_per_second
        self.endpoint_error_splits_meters = endpoint_error_splits_meters

        speed_bucket_bounds = self.speed_bucket_bounds()
        endpoint_error_bucket_bounds = self.epe_bucket_bounds()

        # Bucket by CLASS x SPEED x EPE

        self.per_class_bucketed_error_sum = torch.zeros(
            (len(class_id_to_name_map), len(speed_bucket_bounds),
             len(endpoint_error_bucket_bounds)),
            dtype=torch.float)
        self.per_class_bucketed_error_count = torch.zeros(
            (len(class_id_to_name_map), len(speed_bucket_bounds),
             len(endpoint_error_bucket_bounds)),
            dtype=torch.long)

        self.per_frame_to_per_second_scale_factor = per_frame_to_per_second_scale_factor

    def to(self, device):
        self.per_class_bucketed_error_sum = self.per_class_bucketed_error_sum.to(
            device)
        self.per_class_bucketed_error_count = self.per_class_bucketed_error_count.to(
            device)

    def gather(self, gather_fn):
        per_class_bucketed_error_sum = torch.sum(gather_fn(
            self.per_class_bucketed_error_sum),
                                                 dim=0)
        per_class_bucketed_error_count = torch.sum(gather_fn(
            self.per_class_bucketed_error_count),
                                                   dim=0)
        return per_class_bucketed_error_sum, per_class_bucketed_error_count

    def speed_bucket_bounds(self) -> List[Tuple[float, float]]:
        return list(
            zip(self.speed_bucket_splits_meters_per_second,
                self.speed_bucket_splits_meters_per_second[1:]))

    def epe_bucket_bounds(self) -> List[Tuple[float, float]]:
        return list(
            zip(self.endpoint_error_splits_meters,
                self.endpoint_error_splits_meters[1:]))

    def update_class_error(self, class_id: int, regressed_flow: torch.Tensor,
                           gt_flow: torch.Tensor):

        class_index = self.class_id_to_index_map[class_id]
        endpoint_errors = torch.norm(regressed_flow - gt_flow, dim=1, p=2)
        gt_speeds = torch.norm(gt_flow, dim=1,
                               p=2) * self.per_frame_to_per_second_scale_factor
        # SPEED DISAGGREGATION
        for speed_idx, (lower_speed_bound, upper_speed_bound) in enumerate(
                self.speed_bucket_bounds()):
            speed_mask = (gt_speeds >= lower_speed_bound) & (gt_speeds <
                                                             upper_speed_bound)

            # ENDPOINT ERROR DISAGGREGATION
            for epe_idx, (lower_epe_bound, upper_epe_bound) in enumerate(
                    self.epe_bucket_bounds()):
                endpoint_error_mask = (endpoint_errors >= lower_epe_bound) & (
                    endpoint_errors < upper_epe_bound)

                speed_and_endpoint_error_mask = speed_mask & endpoint_error_mask

                self.per_class_bucketed_error_sum[
                    class_index, speed_idx, epe_idx] += torch.sum(
                        endpoint_errors[speed_and_endpoint_error_mask])
                self.per_class_bucketed_error_count[
                    class_index, speed_idx,
                    epe_idx] += torch.sum(speed_and_endpoint_error_mask)


class ModelWrapper(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
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
            CATEGORY_NAMES, SPEED_BUCKET_SPLITS_METERS_PER_SECOND,
            ENDPOINT_ERROR_SPLITS_METERS)

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
        model_res = self.model(input_batch, **self.val_forward_args)
        output_batch = model_res["forward"]
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

            ground_truth_flow = ground_truth_flowed_pc0_to_pc1 - pc0_pc

            assert pc0_pc.shape == ground_truth_flowed_pc0_to_pc1.shape, f"The input and ground truth pointclouds are not the same shape. {pc0_pc.shape} != {ground_truth_flowed_pc0_to_pc1.shape}"
            assert pc0_pc.shape == regressed_flow.shape, f"The input pc and output flow are not the same shape. {pc0_pc.shape} != {regressed_flow.shape}"

            assert regressed_flow.shape == ground_truth_flow.shape, f"The regressed and ground truth flowed pointclouds are not the same shape."

            # regressed_flowed_pc0_to_pc1 = pc0_pc + regressed_flow
            # if batch_idx % 64 == 0 and minibatch_idx == 0:
            #     self._visualize_regressed_ground_truth_pcs(
            #         pc0_pc, pc1_pc, regressed_flowed_pc0_to_pc1,
            #         ground_truth_flowed_pc0_to_pc1)

            # ======================== Compute Metrics Split By Class ========================

            for cls_id in torch.unique(pc0_pc_class_info):
                cls_mask = (pc0_pc_class_info == cls_id)
                self.metric.update_class_error(cls_id.item(),
                                               regressed_flow[cls_mask],
                                               ground_truth_flow[cls_mask])

    def _save_validation_data(self, save_dict):

        def dict_vals_to_numpy(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    d[k] = dict_vals_to_numpy(v)
                else:
                    d[k] = v.cpu().numpy()
            return d

        save_dict = dict_vals_to_numpy(save_dict)
        save_pickle(f"validation_results/{self.cfg.filename}.pkl", save_dict)

    def validation_epoch_end(self, batch_parts):
        import time
        before_gather = time.time()
        per_class_bucketed_error_sum, per_class_bucketed_error_count = self.metric.gather(
            self.all_gather)

        after_gather = time.time()

        print(
            f"Rank {self.global_rank} gathers done in {after_gather - before_gather}."
        )

        if self.global_rank != 0:
            return {}

        # Per class bucketed error sum and count are CLASS x SPEED x EPE tensors

        # ======================== Compute Full Metric Decomp ========================
        per_class_bucketed_avg = per_class_bucketed_error_sum / per_class_bucketed_error_count

        full_perf_dict = {}

        # PER CLASS
        for cls_idx in range(per_class_bucketed_avg.shape[0]):
            # PER SPEED BUCKET
            for speed_idx, (speed_lower_bound, speed_upper_bound) in zip(
                    range(per_class_bucketed_avg.shape[1]),
                    self.metric.speed_bucket_bounds()):
                # PER EPE BUCKET
                for epe_idx, (epe_lower_bound, epe_upper_bound) in zip(
                        range(per_class_bucketed_avg.shape[2]),
                        self.metric.epe_bucket_bounds()):
                    category_name = self.metric.class_index_to_name_map[
                        cls_idx]
                    bucket_name = f"speed_{speed_lower_bound}-{speed_upper_bound}"
                    epe_bucket_name = f"epe_{epe_lower_bound}-{epe_upper_bound}"
                    error_key = f"val/{category_name}/{bucket_name}/{epe_bucket_name}"
                    value = per_class_bucketed_avg[cls_idx, speed_idx, epe_idx]
                    self.log(error_key,
                             value,
                             sync_dist=False,
                             rank_zero_only=True)
                    full_perf_dict[error_key] = value

        speed_dict = {}
        # ======================== Compute Per Speed Metrics ========================
        for speed_idx, (speed_lower_bound, speed_upper_bound) in zip(
                range(per_class_bucketed_avg.shape[1]),
                self.metric.speed_bucket_bounds()):
            speed_error = per_class_bucketed_error_sum[:, speed_idx, :].sum(
            ) / per_class_bucketed_error_count[:, speed_idx, :].sum()
            bucket_name = f"speed_{speed_lower_bound}-{speed_upper_bound}"
            error_key = f"val/{bucket_name}"
            self.log(error_key,
                     speed_error,
                     sync_dist=False,
                     rank_zero_only=True)
            speed_dict[error_key] = speed_error

        epe_dict = {}
        # ======================== Compute Per EPE Metrics ========================
        for epe_idx, (epe_lower_bound, epe_upper_bound) in zip(
                range(per_class_bucketed_avg.shape[2]),
                self.metric.epe_bucket_bounds()):
            epe_avg_error = per_class_bucketed_error_sum[:, :, epe_idx].sum(
            ) / per_class_bucketed_error_count[:, :, epe_idx].sum()
            epe_error_count = per_class_bucketed_error_count[:, :,
                                                             epe_idx].sum()
            epe_bucket_name = f"epe_{epe_lower_bound}-{epe_upper_bound}"
            error_key = f"val/{epe_bucket_name}"
            self.log(error_key,
                     epe_avg_error,
                     sync_dist=False,
                     rank_zero_only=True)
            epe_dict[error_key] = epe_avg_error
            epe_count_bucket_name = f"epe_count_{epe_lower_bound}-{epe_upper_bound}"
            count_key = f"val/{epe_count_bucket_name}"
            self.log(count_key,
                     epe_error_count,
                     sync_dist=False,
                     rank_zero_only=True)
            epe_dict[count_key] = epe_error_count

        self._save_validation_data({
            "per_class_bucketed_error_sum": per_class_bucketed_error_sum,
            "per_class_bucketed_error_count": per_class_bucketed_error_count,
            "full_perf_dict": full_perf_dict,
            "speed_dict": speed_dict,
            "epe_dict": epe_dict
        })

        ret_dict = {}
        for k, v in full_perf_dict.items():
            ret_dict["full_" + k] = v
        for k, v in speed_dict.items():
            ret_dict["speed_" + k] = v
        for k, v in epe_dict.items():
            ret_dict["epe_" + k] = v

        return ret_dict