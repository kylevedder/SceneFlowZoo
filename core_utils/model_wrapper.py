from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.optim as optim
from bucketed_scene_flow_eval.datastructures import *
import models
from models import ForwardMode
from dataloaders import TorchFullFrameInputSequence, EvalWrapper

from .model_saver import ModelOutSaver, OutputNoSave, OutputSave


def _get_cfg_or_default(cfg, key, default, key_transform=lambda e: e):
    return key_transform(getattr(cfg, key)) if hasattr(cfg, key) else default


class ModelWrapper(pl.LightningModule):
    def __init__(self, cfg, evaluator: EvalWrapper):
        super().__init__()
        self.cfg = cfg
        self.model = models.construct_model(cfg.model.name, cfg.model.args)

        self.lr = _get_cfg_or_default(cfg, "learning_rate", None)
        self.evaluator = evaluator

        self.train_forward_args = _get_cfg_or_default(cfg, "train_forward_args", {})
        self.val_forward_args = _get_cfg_or_default(cfg, "val_forward_args", {})
        self.has_labels = _get_cfg_or_default(cfg, "has_labels", True)

        self.save_output_folder: Optional[Path] = _get_cfg_or_default(
            cfg, "save_output_folder", None, Path
        )

        self.model_out_saver: ModelOutSaver = _get_cfg_or_default(
            cfg, "save_output_folder", OutputNoSave(), OutputSave
        )
        self.cache_validation_outputs: bool = _get_cfg_or_default(
            cfg, "cache_validation_outputs", False
        )
        assert (not self.cache_validation_outputs) or (
            not isinstance(self.model_out_saver, OutputNoSave)
        ), f"Cannot cache outputs with OutputNoSave saver"

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
        self, input_batch: list[TorchFullFrameInputSequence], batch_idx: int
    ) -> dict[str, float]:
        model_res = self.model(
            ForwardMode.TRAIN, input_batch, self.logger, **self.train_forward_args
        )
        loss_res = self.model.loss_fn(input_batch, model_res)
        loss = loss_res.pop("loss")
        self.log("train/loss", loss, on_step=True)
        for k, v in loss_res.items():
            self.log(f"train/{k}", v, on_step=True)
        return {"loss": loss}

    def validation_step(
        self, input_batch: list[TorchFullFrameInputSequence], batch_idx: int
    ) -> None:
        if self.cache_validation_outputs and all(
            [self.model_out_saver.is_saved(sequence) for sequence in input_batch]
        ):
            output_batch = self.model_out_saver.load_saved_batch(input_batch)
        else:
            output_batch = self.model(
                ForwardMode.VAL, input_batch, self.logger, **self.val_forward_args
            )
            self.model_out_saver.save_batch(input_batch, output_batch)

        assert len(output_batch) == len(
            input_batch
        ), f"output minibatch different size than input: {len(output_batch)} != {len(input_batch)}"

        if not self.has_labels:
            return

        self.evaluator.eval_batch(input_batch, output_batch)

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
