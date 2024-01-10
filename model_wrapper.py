import time
from pathlib import Path
from typing import List

import nntime
import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
import tqdm
from bucketed_scene_flow_eval.datastructures import *

import models
from dataloaders import BucketedSceneFlowItem, BucketedSceneFlowOutputItem
from loader_utils import *
from pointclouds import from_fixed_array


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: _to_numpy(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_numpy(v) for v in x]
    return x


class ModelWrapper(pl.LightningModule):
    def __init__(self, cfg, evaluator):
        super().__init__()
        self.cfg = cfg
        self.model = getattr(models, cfg.model.name)(**cfg.model.args)

        if not hasattr(cfg, "is_trainable") or cfg.is_trainable:
            self.loss_fn = getattr(models, cfg.loss_fn.name)(**cfg.loss_fn.args)

        self.lr = cfg.learning_rate
        if hasattr(cfg, "train_forward_args"):
            self.train_forward_args = cfg.train_forward_args
        else:
            self.train_forward_args = {}

        if hasattr(cfg, "val_forward_args"):
            self.val_forward_args = cfg.val_forward_args
        else:
            self.val_forward_args = {}

        self.has_labels = True if not hasattr(cfg, "has_labels") else cfg.has_labels

        self.evaluator = evaluator

    def on_load_checkpoint(self, checkpoint):
        checkpoint_lrs = set()

        for optimizer_state_idx in range(len(checkpoint["optimizer_states"])):
            for param_group_idx in range(
                len(checkpoint["optimizer_states"][optimizer_state_idx]["param_groups"])
            ):
                checkpoint_lrs.add(
                    checkpoint["optimizer_states"][optimizer_state_idx]["param_groups"][
                        param_group_idx
                    ]["lr"]
                )

        # If there are multiple learning rates, or if the learning rate is not the same as the one in the config, reset the optimizer.
        # This is to handle the case where we want to resume training with a different learning rate.

        reset_learning_rate = (len(set(checkpoint_lrs)) != 1) or (
            self.lr != list(checkpoint_lrs)[0]
        )

        if reset_learning_rate:
            print("Resetting learning rate to the one in the config.")
            checkpoint.pop("optimizer_states")
            checkpoint.pop("lr_schedulers")

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return self.optimizer

    def training_step(self, input_batch, batch_idx):
        model_res: List[BucketedSceneFlowOutputItem] = self.model(
            input_batch, **self.train_forward_args
        )
        loss_res = self.loss_fn(input_batch, model_res)
        loss = loss_res.pop("loss")
        self.log("train/loss", loss, on_step=True)
        for k, v in loss_res.items():
            self.log(f"train/{k}", v, on_step=True)
        return {"loss": loss}

    def _visualize_regressed_ground_truth_pcs(
        self, pc0_pc, pc1_pc, regressed_flowed_pc0_to_pc1, ground_truth_flowed_pc0_to_pc1
    ):
        import numpy as np
        import open3d as o3d

        pc0_pc = pc0_pc.cpu().numpy()
        pc1_pc = pc1_pc.cpu().numpy()
        regressed_flowed_pc0_to_pc1 = regressed_flowed_pc0_to_pc1.cpu().numpy()
        ground_truth_flowed_pc0_to_pc1 = ground_truth_flowed_pc0_to_pc1.cpu().numpy()
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
        line_set_points = np.concatenate([pc0_pc, ground_truth_flowed_pc0_to_pc1], axis=0)

        lines = np.array([[i, i + len(ground_truth_flowed_pc0_to_pc1)] for i in range(len(pc0_pc))])
        line_set.points = o3d.utility.Vector3dVector(line_set_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(lines))])
        vis.add_geometry(line_set)

        # Add line set between pc0 and regressed pc1
        line_set = o3d.geometry.LineSet()
        assert len(pc0_pc) == len(
            regressed_flowed_pc0_to_pc1
        ), f"{len(pc0_pc)} != {len(regressed_flowed_pc0_to_pc1)}"
        line_set_points = np.concatenate([pc0_pc, regressed_flowed_pc0_to_pc1], axis=0)

        lines = np.array([[i, i + len(regressed_flowed_pc0_to_pc1)] for i in range(len(pc0_pc))])
        line_set.points = o3d.utility.Vector3dVector(line_set_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(lines))])
        vis.add_geometry(line_set)

        vis.run()

    def validation_step(self, input_batch: List[BucketedSceneFlowItem], batch_idx):
        forward_pass_before = time.time()
        nntime.timer_start(self, "validation_forward")
        start_time = time.time()
        output_batch: List[BucketedSceneFlowOutputItem] = self.model(
            input_batch, **self.val_forward_args
        )
        end_time = time.time()
        nntime.timer_end(self, "validation_forward")
        forward_pass_after = time.time()

        # if self.global_rank != 0:
        #     raise ValueError(
        #         "Validation step should only be run on the master node.")

        if not self.has_labels:
            return

        ################################
        # Save scene trajectory output #
        ################################

        est_pc1_flows_valid_idxes = [
            from_fixed_array(_to_numpy(item.pc0_valid_point_indexes)) for item in output_batch
        ]

        pc1_arrays = [_to_numpy(e.source_pc) for e in input_batch]
        pc_lookup_sizes = np.array([len(pc_array) for pc_array in pc1_arrays])
        pc1_arrays = [
            pc_array[idxes] for pc_array, idxes in zip(pc1_arrays, est_pc1_flows_valid_idxes)
        ]

        est_flows = _to_numpy(output_batch.flow)
        est_flows = [from_fixed_array(flow) for flow in est_flows]

        est_pc2_arrays = [
            pc1_array + est_flow for pc1_array, est_flow in zip(pc1_arrays, est_flows)
        ]

        prepare_data_after = time.time()

        for pc_lookup_size, pc1, pc2, est_pc1_flows_valid_idx, input_elem in zip(
            pc_lookup_sizes,
            pc1_arrays,
            est_pc2_arrays,
            est_pc1_flows_valid_idxes,
            input_batch,
        ):
            # stack pc1 and pc2 together into an N x 2 x 3 array
            stacked_points = np.stack([pc1, pc2], axis=1)
            lookup = EstimatedParticleTrajectories(
                pc_lookup_size, input_elem.gt_trajectories.trajectory_timestamps
            )
            lookup[est_pc1_flows_valid_idx] = (
                stacked_points,
                [0, 1],
                np.zeros((len(pc2), 2), dtype=bool),
            )

            self.evaluator.eval(lookup, input_elem.gt_trajectories, input_elem.query_timestamp)

        loop_data_after = time.time()

        # print("Forward pass time:", end_time - start_time)
        # print("Prepare data time:", prepare_data_after - forward_pass_after)
        # print("Loop data time:", loop_data_after - prepare_data_after)
        # print("Total time: ", loop_data_after - start_time)

    def _save_validation_data(self, save_dict):
        save_pickle(f"validation_results/{self.cfg.filename}.pkl", save_dict)
        try:
            timing_out = f"validation_results/{self.cfg.filename}_timing.csv"
            nntime.export_timings(self, timing_out)
        except AssertionError as e:
            print(f"Could not export timings. Skipping.")

    def _log_validation_metrics(self, validation_result_dict, verbose=True):
        result_full_info = ResultInfo(
            Path(self.cfg.filename).stem, validation_result_dict, full_distance="ALL"
        )
        result_close_info = ResultInfo(
            Path(self.cfg.filename).stem, validation_result_dict, full_distance="CLOSE"
        )
        self.log(
            "val/full/nonmover_epe",
            result_full_info.get_nonmover_point_epe(),
            sync_dist=False,
            rank_zero_only=True,
        )
        self.log(
            "val/full/mover_epe",
            result_full_info.get_mover_point_dynamic_epe(),
            sync_dist=False,
            rank_zero_only=True,
        )
        self.log(
            "val/close/nonmover_epe",
            result_close_info.get_nonmover_point_epe(),
            sync_dist=False,
            rank_zero_only=True,
        )
        self.log(
            "val/close/mover_epe",
            result_close_info.get_mover_point_dynamic_epe(),
            sync_dist=False,
            rank_zero_only=True,
        )

        if verbose:
            print("Validation Results:")
            print(f"Close Mover EPE: {result_close_info.get_mover_point_dynamic_epe()}")
            print(f"Close Nonmover EPE: {result_close_info.get_nonmover_point_epe()}")
            print(f"Full Mover EPE: {result_full_info.get_mover_point_dynamic_epe()}")
            print(f"Full Nonmover EPE: {result_full_info.get_nonmover_point_epe()}")

    def _process_val_files(self):
        assert (
            self.scene_trajectory_output_folder is not None
        ), f"scene_trajectory_output_folder is None"
        pickle_files = sorted(Path(self.scene_trajectory_output_folder).glob("*.pkl"))
        for pickle_file in tqdm.tqdm(pickle_files):
            data = load_pickle(pickle_file, verbose=False)
            dataset_idxes = data["dataset_idxes"]
            query_timestamps_arr = data["query_timestamps"]
            est_pc1_flows_valid_idxes = data["est_pc1_flows_valid_idxes"]

            pc1_points = data["pc1_arrays"]
            est_pc2_points = data["est_pc2_arrays"]

            pc_lookup_sizes = data["pc_lookup_sizes"]

            for (
                dataset_idx,
                query_timestamps,
                pc_lookup_size,
                pc1,
                pc2,
                est_pc1_flows_valid_idx,
            ) in zip(
                dataset_idxes,
                query_timestamps_arr,
                pc_lookup_sizes,
                pc1_points,
                est_pc2_points,
                est_pc1_flows_valid_idxes,
            ):
                # stack pc1 and pc2 together into an N x 2 x 3 array
                stacked_points = np.stack([pc1, pc2], axis=1)
                lookup = EstimatedParticleTrajectories(pc_lookup_size, 2)
                lookup[est_pc1_flows_valid_idx] = (
                    stacked_points,
                    query_timestamps,
                    np.zeros((len(pc2), 2), dtype=bool),
                )
                self.evaluator.eval(dataset_idx, lookup)

    def on_validation_epoch_end(self):
        gathered_evaluator_list = [None for _ in range(torch.distributed.get_world_size())]
        # Get the output from each process
        torch.distributed.all_gather_object(gathered_evaluator_list, self.evaluator)

        if self.global_rank != 0:
            return {}

        # Check that the output from each process is not None
        for idx, output in enumerate(gathered_evaluator_list):
            assert output is not None, f"Output is None for idx {idx}"

        # Merge the outputs from each process into a single object
        gathered_evaluator = self.evaluator.from_evaluator_list(gathered_evaluator_list)
        print("Gathered evaluator of length: ", len(gathered_evaluator))
        return gathered_evaluator.compute_results()
