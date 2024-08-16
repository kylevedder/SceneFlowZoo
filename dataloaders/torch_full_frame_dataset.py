from abc import ABC, abstractmethod

from .dataclasses import TorchFullFrameInputSequence, TorchFullFrameOutputSequence
from .abstract_dataset import BaseDataset, EvalWrapper, TorchEvalWrapper
from .split_dataset import FullFrameDatasetSplit, SplitFullFrameDataset
from pathlib import Path
from typing import Optional, Union

from bucketed_scene_flow_eval.datasets import construct_dataset
from bucketed_scene_flow_eval.interfaces import LoaderType
from dataclasses import dataclass
from bucketed_scene_flow_eval.interfaces import AbstractDataset


class TorchFullFrameDataset(SplitFullFrameDataset):
    def __init__(
        self,
        dataset_name: str,
        root_dir: Path,
        max_pc_points: int = 120000,
        allow_pc_slicing: bool = False,
        set_length: int | None = None,
        split: FullFrameDatasetSplit | dict[str, int] = FullFrameDatasetSplit(0, 1),
        **kwargs,
    ):
        super().__init__(split=split, set_length=set_length)
        self.dataset = construct_dataset(dataset_name, dict(root_dir=root_dir, **kwargs))
        self.max_pc_points = max_pc_points
        self.allow_pc_slicing = allow_pc_slicing

    def evaluator(self) -> TorchEvalWrapper:
        return TorchEvalWrapper(self.dataset)

    def collate_fn(
        self, batch: list[TorchFullFrameInputSequence]
    ) -> list[TorchFullFrameInputSequence]:
        return batch

    def __getitem__(self, split_idx) -> TorchFullFrameInputSequence:
        global_idx = self._get_global_idx(split_idx)
        frame_list = self.dataset[global_idx]
        return TorchFullFrameInputSequence.from_frame_list(
            global_idx,
            frame_list,
            pc_max_len=self.max_pc_points,
            loader_type=self.dataset.loader_type(),
            allow_pc_slicing=self.allow_pc_slicing,
        )
