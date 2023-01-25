import torch
import torch.nn as nn
import torch.optim as optim
import models

import pytorch_lightning as pl
import torchmetrics

CATEGORY_NAMES = {
    -1: (0, 'BACKGROUND'),
    0: (1, 'ANIMAL'),
    1: (2, 'ARTICULATED_BUS'),
    2: (3, 'BICYCLE'),
    3: (4, 'BICYCLIST'),
    4: (5, 'BOLLARD'),
    5: (6, 'BOX_TRUCK'),
    6: (7, 'BUS'),
    7: (8, 'CONSTRUCTION_BARREL'),
    8: (9, 'CONSTRUCTION_CONE'),
    9: (10, 'DOG'),
    10: (11, 'LARGE_VEHICLE'),
    11: (12, 'MESSAGE_BOARD_TRAILER'),
    12: (13, 'MOBILE_PEDESTRIAN_CROSSING_SIGN'),
    13: (14, 'MOTORCYCLE'),
    14: (15, 'MOTORCYCLIST'),
    15: (16, 'OFFICIAL_SIGNALER'),
    16: (17, 'PEDESTRIAN'),
    17: (18, 'RAILED_VEHICLE'),
    18: (19, 'REGULAR_VEHICLE'),
    19: (20, 'SCHOOL_BUS'),
    20: (21, 'SIGN'),
    21: (22, 'STOP_SIGN'),
    22: (23, 'STROLLER'),
    23: (24, 'TRAFFIC_LIGHT_TRAILER'),
    24: (25, 'TRUCK'),
    25: (26, 'TRUCK_CAB'),
    26: (27, 'VEHICULAR_TRAILER'),
    27: (28, 'WHEELCHAIR'),
    28: (29, 'WHEELED_DEVICE'),
    29: (30, 'WHEELED_RIDER')
}


class EndpointDistanceMetrics(torchmetrics.Metric):

    def __init__(self, name="my_metric", dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("per_class_total_error_sum",
                       default=torch.zeros((len(CATEGORY_NAMES))),
                       dist_reduce_fx="sum")
        self.add_state("per_class_total_error_count",
                       default=torch.zeros((len(CATEGORY_NAMES)),
                                           dtype=torch.long),
                       dist_reduce_fx="sum")

    def update(self, endpoint_error, endpoint_count):
        self.per_class_total_error_sum += endpoint_error
        self.per_class_total_error_count += endpoint_count

    def compute(self):
        return self.per_class_total_error_sum / self.per_class_total_error_count


class ModelWrapper(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.model = getattr(models, cfg.model.name)(**cfg.model.args)
        self.loss_fn = getattr(models, cfg.loss_fn.name)(**cfg.loss_fn.args)
        self.lr = cfg.learning_rate

        self.metric = EndpointDistanceMetrics()

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

        per_class_total_error_count = torch.zeros((len(CATEGORY_NAMES)),
                                                  dtype=torch.long,
                                                  device=self.device)
        per_class_total_error_sum = torch.zeros((len(CATEGORY_NAMES)),
                                                device=self.device)

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
                cls_mask = pc0_pc_class_info == cls_id
                cls_distance_array = distance_array[cls_mask]
                cls_avg_distance = torch.mean(cls_distance_array)
                cls_idx, category_name = CATEGORY_NAMES[int(cls_id.item())]
                per_class_total_error_sum[cls_idx] += torch.sum(
                    cls_distance_array)
                per_class_total_error_count[
                    cls_idx] += cls_distance_array.shape[0]

        self.metric(per_class_total_error_sum, per_class_total_error_count)

    def validation_epoch_end(self, batch_parts):
        print("Validation epoch end start.")
        metric_result = self.metric.compute()
        print("Validation epoch end compute() done.")

        reverse_category_names = {
            cls_idx: name
            for _, (cls_idx, name) in CATEGORY_NAMES.items()
        }

        perf_dict = {}
        for cls_idx, cls_avg_error in enumerate(metric_result):
            category_name = reverse_category_names[cls_idx]
            perf_dict[category_name] = cls_avg_error.item()
            self.log(f"val/{category_name}",
                     cls_avg_error,
                     on_epoch=True,
                     sync_dist=False, 
                     rank_zero_only=True)

        print("Validation epoch end end.")

        return perf_dict