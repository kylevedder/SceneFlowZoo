import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import models
from pathlib import Path
from pointclouds import from_fixed_array

import pytorch_lightning as pl
import torchmetrics
from typing import Dict, List, Tuple
from loader_utils import *
import nntime
import time
import multiprocessing as mp
from functools import partial


class EndpointDistanceMetricRawTorch():

    def __init__(self,
                 class_id_to_name_map: Dict[int, str],
                 speed_bucket_splits_meters_per_second: List[float],
                 endpoint_error_splits_meters: List[float],
                 close_object_threshold_meters: float = 35.0,
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

        # Bucket by IS_CLOSE x CLASS x SPEED x EPE

        self.per_class_bucketed_error_sum = torch.zeros(
            (2, len(class_id_to_name_map), len(speed_bucket_bounds),
             len(endpoint_error_bucket_bounds)),
            dtype=torch.float)
        self.per_class_bucketed_error_count = torch.zeros(
            (2, len(class_id_to_name_map), len(speed_bucket_bounds),
             len(endpoint_error_bucket_bounds)),
            dtype=torch.long)

        self.close_object_threshold_meters = close_object_threshold_meters
        self.per_frame_to_per_second_scale_factor = per_frame_to_per_second_scale_factor

        self.total_forward_time = torch.tensor(0.0)
        self.total_forward_count = torch.tensor(0, dtype=torch.long)

    def to(self, device):
        self.per_class_bucketed_error_sum = self.per_class_bucketed_error_sum.to(
            device)
        self.per_class_bucketed_error_count = self.per_class_bucketed_error_count.to(
            device)
        self.total_forward_time = self.total_forward_time.to(device)
        self.total_forward_count = self.total_forward_count.to(device)

    def gather(self, gather_fn):
        per_class_bucketed_error_sum = torch.sum(gather_fn(
            self.per_class_bucketed_error_sum),
                                                 dim=0)
        per_class_bucketed_error_count = torch.sum(gather_fn(
            self.per_class_bucketed_error_count),
                                                   dim=0)
        total_forward_time = torch.sum(gather_fn(self.total_forward_time),
                                       dim=0)
        total_forward_count = torch.sum(gather_fn(self.total_forward_count),
                                        dim=0)
        return per_class_bucketed_error_sum, per_class_bucketed_error_count, total_forward_time, total_forward_count

    def speed_bucket_bounds(self) -> List[Tuple[float, float]]:
        return list(
            zip(self.speed_bucket_splits_meters_per_second,
                self.speed_bucket_splits_meters_per_second[1:]))

    def epe_bucket_bounds(self) -> List[Tuple[float, float]]:
        return list(
            zip(self.endpoint_error_splits_meters,
                self.endpoint_error_splits_meters[1:]))

    def update_class_error(self, pc: torch.Tensor, class_id: int,
                           regressed_flow: torch.Tensor,
                           gt_flow: torch.Tensor):

        assert regressed_flow.shape == gt_flow.shape, f"Shapes do not match: {regressed_flow.shape} vs {gt_flow.shape}"
        assert regressed_flow.shape[0] == pc.shape[
            0], f"Shapes do not match: {regressed_flow.shape[0]} vs {pc.shape[0]}"
        assert pc.shape[
            1] == 3, f"Shapes do not match: {regressed_flow.shape[1]} vs 3"

        # L_\infty norm the XY coordinates needs to be within the close object threshold.
        xy_points = pc[:, :2]
        point_xy_distances = torch.norm(xy_points, dim=1, p=np.inf)
        is_close_mask = point_xy_distances <= self.close_object_threshold_meters

        class_index = self.class_id_to_index_map[class_id]
        endpoint_errors = torch.norm(regressed_flow - gt_flow, dim=1, p=2)

        gt_speeds = torch.norm(gt_flow, dim=1,
                               p=2) * self.per_frame_to_per_second_scale_factor

        # IS CLOSE DISAGGREGATION
        for close_mask_idx, close_mask in enumerate(
            [is_close_mask, ~is_close_mask]):
            # SPEED DISAGGREGATION
            for speed_idx, (lower_speed_bound, upper_speed_bound) in enumerate(
                    self.speed_bucket_bounds()):
                speed_mask = (gt_speeds >= lower_speed_bound) & (
                    gt_speeds < upper_speed_bound)

                # ENDPOINT ERROR DISAGGREGATION
                for epe_idx, (lower_epe_bound, upper_epe_bound) in enumerate(
                        self.epe_bucket_bounds()):
                    endpoint_error_mask = (
                        endpoint_errors >=
                        lower_epe_bound) & (endpoint_errors < upper_epe_bound)
                    total_mask = close_mask & speed_mask & endpoint_error_mask

                    self.per_class_bucketed_error_sum[
                        close_mask_idx, class_index, speed_idx,
                        epe_idx] += torch.sum(endpoint_errors[total_mask])
                    self.per_class_bucketed_error_count[
                        close_mask_idx, class_index, speed_idx,
                        epe_idx] += torch.sum(total_mask)

    def update_runtime(self, run_time: float, run_count: int):
        self.total_forward_time += run_time
        self.total_forward_count += run_count

    def reset(self):
        self.per_class_bucketed_error_sum.zero_()
        self.per_class_bucketed_error_count.zero_()
        self.total_forward_time.zero_()
        self.total_forward_count.zero_()


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

        self.save_output_folder = None if not hasattr(
            cfg, "save_output_folder") else cfg.save_output_folder
        self.save_pool = None if self.save_output_folder is None else mp.Pool(
            mp.cpu_count())

        self.metric = EndpointDistanceMetricRawTorch(
            CATEGORY_ID_TO_NAME, SPEED_BUCKET_SPLITS_METERS_PER_SECOND,
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

    def _save_output(self, input_batch, output_batch, batch_idx, delta_time):

        def _to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            if isinstance(x, dict):
                return {k: _to_numpy(v) for k, v in x.items()}
            if isinstance(x, list):
                return [_to_numpy(v) for v in x]
            return x

        log_ids = np.transpose(_to_numpy(input_batch["log_ids"]))
        log_idxes = np.transpose(_to_numpy(input_batch["log_idxes"]))
        pc_array_stack = _to_numpy(input_batch['pc_array_stack'])
        gt_flows = input_batch['flowed_pc_array_stack'] - input_batch[
            'pc_array_stack']
        est_flows = output_batch['flow']
        est_pc1_flows_valid_idxes = output_batch['pc0_valid_point_idxes']
        est_pc2_flows_valid_idxes = output_batch['pc1_valid_point_idxes']

        save_list = []

        for pc_arrays, log_ids, log_idxs, gt_flows, est_flow, est_pc1_flows_valid_idx, est_pc2_flows_valid_idx in zip(
                pc_array_stack, log_ids, log_idxes, gt_flows, est_flows,
                est_pc1_flows_valid_idxes, est_pc2_flows_valid_idxes):
            pc1 = _to_numpy(from_fixed_array(pc_arrays[0]))
            pc2 = _to_numpy(from_fixed_array(pc_arrays[1]))
            log_id = log_ids[0]
            log_idx = log_idxs[0]
            gt_flow = _to_numpy(from_fixed_array(gt_flows[0]))
            est_flow = _to_numpy(from_fixed_array(est_flow))
            est_pc1_flows_valid_idx = _to_numpy(est_pc1_flows_valid_idx)
            est_pc2_flows_valid_idx = _to_numpy(est_pc2_flows_valid_idx)
            expanded_save_file = Path(
                self.save_output_folder) / log_id / f"{log_idx:06d}.npy"
            standard_save_file = Path(
                self.save_output_folder) / log_id / f"{log_idx:06d}.npz"

            expanded_data_dict = {
                "pc1": pc1[est_pc1_flows_valid_idx],
                "pc2": pc2[est_pc2_flows_valid_idx],
                "gt_flow": gt_flow[est_pc1_flows_valid_idx],
                "est_flow": est_flow,
                "pc1_flows_valid_idx": est_pc1_flows_valid_idx,
                "pc2_flows_valid_idx": est_pc2_flows_valid_idx,
            }

            standard_data_dict = {
                'flow': est_flow,
                'valid_idxes': est_pc1_flows_valid_idx,
                'delta_time': delta_time
            }

            save_list.append((expanded_save_file, expanded_data_dict))
            save_list.append((standard_save_file, standard_data_dict))

        self.save_pool.starmap(partial(save_by_extension, verbose=False),
                               save_list)

    def validation_step(self, input_batch, batch_idx):
        nntime.timer_start(self, "validation_forward")
        start_time = time.time()
        model_res = self.model(input_batch, **self.val_forward_args)
        end_time = time.time()
        nntime.timer_end(self, "validation_forward")
        output_batch = model_res["forward"]
        self.metric.to(self.device)

        if self.save_output_folder is not None:
            self._save_output(input_batch, output_batch, batch_idx,
                              end_time - start_time)

        if not self.has_labels:
            return

        self.metric.update_runtime(output_batch["batch_delta_time"],
                                   len(input_batch["pc_array_stack"]))

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
                self.metric.update_class_error(pc0_pc[cls_mask], cls_id.item(),
                                               regressed_flow[cls_mask],
                                               ground_truth_flow[cls_mask])

    def _dict_vals_to_numpy(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = self._dict_vals_to_numpy(v)
            else:
                d[k] = v.cpu().numpy()
        return d

    def _save_validation_data(self, save_dict):
        save_pickle(f"validation_results/{self.cfg.filename}.pkl", save_dict)
        try:
            timing_out = f"validation_results/{self.cfg.filename}_timing.csv"
            nntime.export_timings(self, timing_out)
        except AssertionError as e:
            print(f"Could not export timings. Skipping.")

    def _log_validation_metrics(self, validation_result_dict, verbose=True):
        result_full_info = ResultInfo(Path(self.cfg.filename).stem,
                                      validation_result_dict,
                                      full_distance='ALL')
        result_close_info = ResultInfo(Path(self.cfg.filename).stem,
                                       validation_result_dict,
                                       full_distance='CLOSE')
        self.log("val/full/nonmover_epe",
                 result_full_info.get_nonmover_point_epe(),
                 sync_dist=False,
                 rank_zero_only=True)
        self.log("val/full/mover_epe",
                 result_full_info.get_mover_point_dynamic_epe(),
                 sync_dist=False,
                 rank_zero_only=True)
        self.log("val/close/nonmover_epe",
                 result_close_info.get_nonmover_point_epe(),
                 sync_dist=False,
                 rank_zero_only=True)
        self.log("val/close/mover_epe",
                 result_close_info.get_mover_point_dynamic_epe(),
                 sync_dist=False,
                 rank_zero_only=True)

        if verbose:
            print("Validation Results:")
            print(
                f"Close Mover EPE: {result_close_info.get_mover_point_dynamic_epe()}"
            )
            print(
                f"Close Nonmover EPE: {result_close_info.get_nonmover_point_epe()}"
            )
            print(
                f"Full Mover EPE: {result_full_info.get_mover_point_dynamic_epe()}"
            )
            print(
                f"Full Nonmover EPE: {result_full_info.get_nonmover_point_epe()}"
            )

    def validation_epoch_end(self, batch_parts):
        import time
        before_gather = time.time()

        # These are copies of the metric values on each rank.
        per_class_bucketed_error_sum, per_class_bucketed_error_count, total_forward_time, total_forward_count = self.metric.gather(
            self.all_gather)

        after_gather = time.time()

        print(
            f"Rank {self.global_rank} gathers done in {after_gather - before_gather}."
        )

        # Reset the metric for the next epoch. We have to do this on each rank, and because we are using
        # copies of the metric values above, we don't have to worry about over-writing the values.
        self.metric.reset()

        if self.global_rank != 0:
            return {}

        validation_result_dict = {
            "per_class_bucketed_error_sum": per_class_bucketed_error_sum,
            "per_class_bucketed_error_count": per_class_bucketed_error_count,
            "average_forward_time": total_forward_time / total_forward_count
        }

        validation_result_dict = self._dict_vals_to_numpy(
            validation_result_dict)

        self._log_validation_metrics(validation_result_dict)

        self._save_validation_data(validation_result_dict)

        return {}