import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.optim as optim
from bucketed_scene_flow_eval.datastructures import *

import models
from dataloaders import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence, EvalWrapper
from loader_utils import *


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: _to_numpy(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_numpy(v) for v in x]
    return x


class ModelWrapper(pl.LightningModule):
    def __init__(self, cfg, evaluator: EvalWrapper):
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
        self.save_output_folder: Optional[Path] = (
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

    def training_step(
        self, input_batch: list[BucketedSceneFlowInputSequence], batch_idx: int
    ) -> dict[str, float]:
        model_res: list[BucketedSceneFlowOutputSequence] = self.model(
            input_batch, **self.train_forward_args
        )
        loss_res = self.loss_fn(input_batch, model_res)
        loss = loss_res.pop("loss")
        self.log("train/loss", loss, on_step=True)
        for k, v in loss_res.items():
            self.log(f"train/{k}", v, on_step=True)
        return {"loss": loss}

    def _save_output_flat_structure(
        self, input: BucketedSceneFlowInputSequence, output: BucketedSceneFlowOutputSequence
    ):
        """
        Save each flow as a single result.

        This assumes that the output is length is 1.

        The content of the feather file is a dataframe with the following columns:
         - is_valid
         - flow_tx_m
         - flow_ty_m
         - flow_tz_m

        The feather file is named {save_dir} / {input_length} / {dataset_log_id} / {dataset_idx}.feather
        """
        ego_flows = output.to_ego_lidar_flow_list()
        assert len(ego_flows) == 1, f"Expected a single ego flow, but got {len(ego_flows)}"
        ego_flow = ego_flows[0]
        output_df = pd.DataFrame(
            {
                "is_valid": ego_flow.mask,
                "flow_tx_m": ego_flow.full_flow[:, 0],
                "flow_ty_m": ego_flow.full_flow[:, 1],
                "flow_tz_m": ego_flow.full_flow[:, 2],
            }
        )
        assert self.save_output_folder is not None, "self.save_output_folder is None"
        save_path = (
            Path(self.save_output_folder)
            / f"sequence_len_{len(input):03d}"
            / f"{input.sequence_log_id}"
            / f"{input.dataset_idx:010d}.feather"
        )
        save_feather(
            save_path,
            output_df,
            verbose=False,
        )

    def _save_output_sequence(
        self, input: BucketedSceneFlowInputSequence, output: BucketedSceneFlowOutputSequence
    ):
        if len(output.ego_flows) == 1:
            self._save_output_flat_structure(input, output)
            return
        else:
            raise NotImplementedError(
                f"Saving multiple ego flows (given {len(output.ego_flows)}) is not yet implemented."
            )

    def _save_output_batch(
        self,
        input_batch: list[BucketedSceneFlowInputSequence],
        output_batch: list[BucketedSceneFlowOutputSequence],
    ):
        for input_elem, output_elem in zip(input_batch, output_batch):
            self._save_output_sequence(input_elem, output_elem)

    def validation_step(
        self, input_batch: list[BucketedSceneFlowInputSequence], batch_idx: int
    ) -> None:
        output_batch: list[BucketedSceneFlowOutputSequence] = self.model(
            input_batch, **self.val_forward_args
        )

        assert len(output_batch) == len(
            input_batch
        ), f"output minibatch different size than input: {len(output_batch)} != {len(input_batch)}"

        if self.save_output_folder is not None:
            self._save_output_batch(input_batch, output_batch)

        if not self.has_labels:
            return

        ################################
        # Save scene trajectory output #
        ################################

        for input_elem, output_elem in zip(input_batch, output_batch):
            self.evaluator.eval(input_elem, output_elem)

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

        gathered_evaluator_list: list[EvalWrapper] = [
            None for _ in range(torch.distributed.get_world_size())
        ]
        # Get the output from each process
        torch.distributed.all_gather_object(gathered_evaluator_list, self.evaluator)

        if self.global_rank != 0:
            return {}

        # Check that the output from each process is not None
        for idx, output in enumerate(gathered_evaluator_list):
            assert output is not None, f"Output is None for idx {idx}"

        # Merge the outputs from each process into a single object
        gathered_evaluator = sum(e.evaluator for e in gathered_evaluator_list)
        print("Gathered evaluator of length: ", len(gathered_evaluator))
        return gathered_evaluator.compute_results()
