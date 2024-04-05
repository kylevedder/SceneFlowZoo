from abc import ABC, abstractmethod

from .dataclasses import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from bucketed_scene_flow_eval.datasets import construct_dataset
from bucketed_scene_flow_eval.datastructures import SE3, TimeSyncedSceneFlowFrame, EgoLidarFlow
from bucketed_scene_flow_eval.interfaces import LoaderType, AbstractDataset


class EvalWrapper:

    def __init__(self, dataset: AbstractDataset):
        self.dataset = dataset
        self.evaluator = dataset.evaluator()

    def _eval_causal(
        self,
        gt_frame_list: list[TimeSyncedSceneFlowFrame],
        pred_list: list[EgoLidarFlow],
    ):
        # Expect that the output is a single ego flow.
        assert (
            len(pred_list) == 1
        ), f"As a causal loader, expected a single ego flow for the final frame pair, but instead got {len(pred_list)} flows."
        assert (
            len(gt_frame_list) >= 2
        ), f"Expected at least two frames, but got {len(gt_frame_list)}."

        gt = gt_frame_list[-2]
        pred = pred_list[0]

        self.evaluator.eval(pred, gt)

    def _eval_non_causal(
        self,
        gt_frame_list: list[TimeSyncedSceneFlowFrame],
        pred_list: list[EgoLidarFlow],
    ):
        # Expect that the output is a list of ego flows, one for each frame pair.
        assert (
            len(pred_list) == len(gt_frame_list) - 1
        ), f"As a non-causal loader, expected a single ego flow for each frame pair, but instead got {len(pred_list)} flows for {len(gt_frame_list) - 1} frame pairs."

        for gt, pred in zip(gt_frame_list[:-1], pred_list):
            self.evaluator.eval(predictions=pred, gt=gt)

    def eval(self, input: BucketedSceneFlowInputSequence, output: BucketedSceneFlowOutputSequence):
        gt_frame_list = self.dataset[input.dataset_idx]
        pred_list = output.to_ego_lidar_flow_list()
        if self.dataset.loader_type() == LoaderType.CAUSAL:
            self._eval_causal(gt_frame_list, pred_list)
        elif self.dataset.loader_type() == LoaderType.NON_CAUSAL:
            self._eval_non_causal(gt_frame_list, pred_list)
        else:
            raise ValueError(f"Unknown loader type: {self.dataset.loader_type()}")

    def eval_batch(
        self,
        inputs: List[BucketedSceneFlowInputSequence],
        outputs: List[BucketedSceneFlowOutputSequence],
    ):
        assert len(inputs) == len(
            outputs
        ), f"Expected the same number of inputs and outputs, but got {len(inputs)} inputs and {len(outputs)} outputs."
        for input, output in zip(inputs, outputs):
            self.eval(input, output)

    def compute_results(self, save_results: bool = True) -> dict:
        return self.evaluator.compute_results(save_results=save_results)


class BucketedSceneFlowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        root_dir: Path,
        max_pc_points: int = 120000,
        set_length: Optional[int] = None,
        **kwargs,
    ):
        self.dataset = construct_dataset(dataset_name, dict(root_dir=root_dir, **kwargs))
        self.max_pc_points = max_pc_points
        self.set_length = set_length

    def __len__(self):
        if self.set_length is not None:
            return self.set_length
        return len(self.dataset)

    def evaluator(self) -> EvalWrapper:
        return EvalWrapper(self.dataset)

    def collate_fn(
        self, batch: list[BucketedSceneFlowInputSequence]
    ) -> list[BucketedSceneFlowInputSequence]:
        return batch

    def __getitem__(self, idx) -> BucketedSceneFlowInputSequence:
        assert isinstance(idx, int), f"Index must be an integer. Got {type(idx)}."
        assert 0 <= idx < len(self), "Index out of range."
        frame_list = self.dataset[idx]
        return BucketedSceneFlowInputSequence.from_frame_list(
            idx,
            frame_list,
            pc_max_len=self.max_pc_points,
            loader_type=self.dataset.loader_type(),
        )
