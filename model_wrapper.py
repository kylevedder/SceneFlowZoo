import torch
import torch.nn as nn
import torch.optim as optim
import models

import pytorch_lightning as pl
import torchmetrics
from typing import Dict

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


class EndpointDistanceMetricRawTorch():

    def __init__(self, class_id_to_name_map: Dict[int, str]):
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

    def to(self, device):
        self.per_class_total_error_sum = self.per_class_total_error_sum.to(
            device)
        self.per_class_total_error_count = self.per_class_total_error_count.to(
            device)

    def update(self, class_id: int, endpoint_error, endpoint_count: int):
        class_index = self.class_id_to_index_map[class_id]
        self.per_class_total_error_sum[class_index] += endpoint_error
        self.per_class_total_error_count[class_index] += endpoint_count

    def compute(self):
        return self.per_class_total_error_sum / self.per_class_total_error_count


class ModelWrapper(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.model = getattr(models, cfg.model.name)(**cfg.model.args)
        self.loss_fn = getattr(models, cfg.loss_fn.name)(**cfg.loss_fn.args)
        self.lr = cfg.learning_rate

        self.metric = EndpointDistanceMetricRawTorch(CATEGORY_NAMES)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, input_batch, batch_idx):
        model_res = self.model(input_batch)
        loss = self.loss_fn(model_res)
        self.log("train/loss", loss, on_step=True)
        return {"loss": loss}

    def validation_step(self, input_batch, batch_idx):
        output_batch = self.model(input_batch)
        self.metric.to(self.device)

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

            # Compute average error per class.
            for cls_id in torch.unique(pc0_pc_class_info):
                cls_mask = (pc0_pc_class_info == cls_id)
                cls_distance_array = distance_array[cls_mask]
                total_distance = torch.sum(cls_distance_array)
                self.metric.update(cls_id.item(), total_distance,
                                   cls_distance_array.shape[0])

    def validation_epoch_end(self, batch_parts):
        print("Starting gather", flush=True)
        import time
        before_gather = time.time()
        world_per_class_total_error_sum = self.all_gather(
            self.metric.per_class_total_error_sum)
        world_per_class_total_error_count = self.all_gather(
            self.metric.per_class_total_error_count)
        after_gather = time.time()

        print(
            f"Rank {self.global_rank} gathers done in {after_gather - before_gather}."
        )

        if self.global_rank != 0:
            return {}

        # Sum the per-GPU metrics across the world axis.
        per_class_total_error_sum = torch.sum(
            world_per_class_total_error_sum, dim=0)
        per_class_total_error_count = torch.sum(
            world_per_class_total_error_count, dim=0)

        cls_avg_errors = per_class_total_error_sum / per_class_total_error_count

        perf_dict = {}
        for cls_idx, cls_avg_error in enumerate(cls_avg_errors):
            category_name = self.metric.class_index_to_name_map[cls_idx]
            perf_dict[category_name] = cls_avg_error.item()
            self.log(f"val/{category_name}",
                     cls_avg_error,
                     sync_dist=False,
                     rank_zero_only=True)

        print("Validation epoch end end.")

        return perf_dict