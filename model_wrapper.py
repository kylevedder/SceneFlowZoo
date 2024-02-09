import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.optim as optim
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

        if cfg.model.name == "CacheWrapper":
            self.model = getattr(models, cfg.model.name)(cfg)
        else:
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
        self.save_output_folder = (
            None if not hasattr(cfg, "save_output_folder") else cfg.save_output_folder
        )

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

    def training_step(self, input_batch: list[BucketedSceneFlowItem], batch_idx: int) -> dict[str, float]:
        model_res: list[BucketedSceneFlowOutputItem] = self.model(
            input_batch, **self.train_forward_args
        )
        loss_res = self.loss_fn(input_batch, model_res)
        loss = loss_res.pop("loss")
        self.log("train/loss", loss, on_step=True)
        for k, v in loss_res.items():
            self.log(f"train/{k}", v, on_step=True)
        return {"loss": loss}

    def _save_output(
        self,
        input_batch: list[BucketedSceneFlowItem],
        output_batch: list[BucketedSceneFlowOutputItem],
    ):
        """
        Save each element in the batch as a separate feather file.

        The content of the feather file is a dataframe with the following columns:
         - is_valid
         - flow_tx_m
         - flow_ty_m
         - flow_tz_m

        The feather file is named {save_dir} / {dataset_log_id} / {dataset_idx}.feather
        """
        for input_elem, output_elem in zip(input_batch, output_batch):
            raw_source_pc = _to_numpy(input_elem.raw_source_pc)
            raw_source_pc_mask = _to_numpy(input_elem.raw_source_pc_mask)
            valid_pc0_mask = _to_numpy(output_elem.pc0_valid_point_mask)

            full_flow_buffer = np.zeros_like(raw_source_pc)  # Default to zero vector
            full_is_valid_buffer = np.zeros_like(raw_source_pc_mask)  # Default to False

            cropped_flow_buffer = full_flow_buffer[raw_source_pc_mask].copy()
            cropped_is_valid_buffer = full_is_valid_buffer[raw_source_pc_mask].copy()

            assert cropped_flow_buffer.shape[0] == valid_pc0_mask.shape[0], (
                f"Flow and valid mask shapes do not match: {cropped_flow_buffer.shape} != {valid_pc0_mask.shape}"
            )
            cropped_flow_buffer = _to_numpy(output_elem.flow)
            cropped_is_valid_buffer[valid_pc0_mask] = True

            full_flow_buffer[raw_source_pc_mask] = cropped_flow_buffer
            full_is_valid_buffer[raw_source_pc_mask] = cropped_is_valid_buffer

            assert full_is_valid_buffer.sum() == valid_pc0_mask.sum(), f"Invalid is_valid_buffer"
            output_df = pd.DataFrame(
                {
                    "is_valid": full_is_valid_buffer,
                    "flow_tx_m": full_flow_buffer[:, 0],
                    "flow_ty_m": full_flow_buffer[:, 1],
                    "flow_tz_m": full_flow_buffer[:, 2],
                }
            )
            save_feather(
                Path(self.save_output_folder)
                / f"{input_elem.dataset_log_id}/{input_elem.dataset_idx:010d}.feather",
                output_df,
                verbose=False,
            )

    def validation_step(self, input_batch: list[BucketedSceneFlowItem], batch_idx: int) -> None:
        forward_pass_before = time.time()
        start_time = time.time()
        output_batch: list[BucketedSceneFlowOutputItem] = self.model(
            input_batch, **self.val_forward_args
        )
        end_time = time.time()
        forward_pass_after = time.time()

        assert len(output_batch) == len(
            input_batch
        ), f"output minibatch different size than input: {len(output_batch)} != {len(input_batch)}"

        if self.save_output_folder is not None:
            self._save_output(input_batch, output_batch)

        if not self.has_labels:
            return

        ################################
        # Save scene trajectory output #
        ################################

        for input_elem, output_elem in zip(input_batch, output_batch):
            # The valid input pc should be the same size as the raw output pc
            assert input_elem.source_pc.shape == output_elem.pc0_points.shape, (
                f"Input and output shapes do not match: {input_elem.source_pc.shape} != {output_elem.pc0_points.shape}"
            )

            output_valid_mask = _to_numpy(output_elem.pc0_valid_point_mask)
            pc0 = _to_numpy(output_elem.pc0_points)[output_valid_mask]
            flowed_pc0 = _to_numpy(output_elem.pc0_warped_points)[output_valid_mask]


            masked_stacked_points = np.stack([pc0, flowed_pc0], axis=1)

            raw_valid_mask = _to_numpy(input_elem.raw_source_pc_mask)
            raw_valid_mask[raw_valid_mask] = output_valid_mask

            lookup = EstimatedPointFlow(
                input_elem.gt_trajectories.num_entries,
                input_elem.gt_trajectories.trajectory_timestamps,
            )
            lookup[raw_valid_mask] = masked_stacked_points

            self.evaluator.eval(lookup, input_elem.gt_trajectories, input_elem.query_timestamp)

        loop_data_after = time.time()

        # print("Forward pass time:", end_time - start_time)
        # print("Prepare data time:", prepare_data_after - forward_pass_after)
        # print("Loop data time:", loop_data_after - prepare_data_after)
        # print("Total time: ", loop_data_after - start_time)

    def _save_validation_data(self, save_dict):
        save_pickle(f"validation_results/{self.cfg.filename}.pkl", save_dict)

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

    def on_validation_epoch_end(self):
        if not self.has_labels:
            return {}

        gathered_evaluator_list = [None for _ in range(torch.distributed.get_world_size())]
        # Get the output from each process
        torch.distributed.all_gather_object(gathered_evaluator_list, self.evaluator)

        if self.global_rank != 0:
            return {}

        # Check that the output from each process is not None
        for idx, output in enumerate(gathered_evaluator_list):
            assert output is not None, f"Output is None for idx {idx}"

        # Merge the outputs from each process into a single object
        gathered_evaluator = sum(gathered_evaluator_list)
        print("Gathered evaluator of length: ", len(gathered_evaluator))
        return gathered_evaluator.compute_results()
