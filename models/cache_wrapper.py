from typing import Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import models
from dataloaders import BucketedSceneFlowItem, BucketedSceneFlowOutputItem

from loader_utils.loaders import load_feather

class CacheWrapper(nn.Module):
    """A wrapper class that loads inference results from disk if they exist.

    When a runtime optimization method (such as NSFP) is run on a cluster (SLURM) the job can fail.
    In this scenario, it is helpful to restart the job and load the saved inference results from disk instead of re-running the optimization.
    Inference results are saved by the lightning model wrapper in `model_wrapper.py` and are used for psuedo labels.
    This wrapper module is really only intended to be used with runtime optimization methods (such as NSFP).
    Feedforward methods don't default to saving their inference results to disk, and can be cheaply re-run.
    """
    def __init__(self, cfg) -> None:
        """Initialize the cache wrapper. Expects the full training config.

        Within the training config, wrap models with the following format:
        model = dict(name="CacheWrapper",
                     args=dict(model="YourModelName",
                                args=dict(MODEL_ARG=1, MODEL_ARG=[2, 3, 4])))
        """
        super().__init__()
        # Construct the inner model. cfg.model refers to this wrapper, and cfg.model.args.model to the inner one
        self.model = getattr(models, cfg.model.args.model)(**cfg.model.args.args)

        if hasattr(cfg, "train_forward_args"):
            self.train_forward_args = cfg.train_forward_args
        else:
            self.train_forward_args = {}

        self.save_output_folder: Path | None = (
            None if not hasattr(cfg, "save_output_folder") else Path(cfg.save_output_folder)
        )

    def forward(self, input_batch: list[BucketedSceneFlowItem]) -> list[BucketedSceneFlowOutputItem]:
        if self.save_output_folder and self.save_output_folder.exists():
            output_batch: list[BucketedSceneFlowOutputItem] = []
            for input_elem in input_batch:
                frame_output_path = Path(self.save_output_folder)/f"{input_elem.dataset_log_id}/{input_elem.dataset_idx:010d}.feather"
                output_pandas_dataframe = load_feather(frame_output_path)
                # convert output frame to bucketed scene flow output item
                xs = output_pandas_dataframe["flow_tx_m"].values
                ys = output_pandas_dataframe["flow_ty_m"].values
                zs = output_pandas_dataframe["flow_tz_m"].values
                flow = np.stack([xs, ys, zs], axis=1)
                is_valid_arr = output_pandas_dataframe["is_valid"].values
                cropped_flow = torch.Tensor(flow)[input_elem.raw_source_pc_mask]
                cropped_flow = cropped_flow.to(input_elem.source_pc.device)
                cropped_mask = torch.Tensor(is_valid_arr)[input_elem.raw_source_pc_mask]
                cropped_mask = cropped_mask.to(dtype=torch.bool)
                output_item = BucketedSceneFlowOutputItem(
                    flow=cropped_flow,  # type: ignore[arg-type]
                    pc0_points=input_elem.source_pc,
                    pc0_valid_point_mask=cropped_mask,  # type: ignore[arg-type]
                    pc0_warped_points=input_elem.source_pc + cropped_flow,  # type: ignore[arg-type]
                )
                output_item = output_item.to(input_elem.source_pc.device)
                output_batch.append(output_item)
        else:
            output_batch = self.model(input_batch, **self.train_forward_args)
        return output_batch
