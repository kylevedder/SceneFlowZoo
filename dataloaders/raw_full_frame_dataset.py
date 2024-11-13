from .dataclasses import (
    BaseInputSequence,
    BaseOutputSequence,
)
from .abstract_dataset import EvalWrapper
from .split_dataset import FullFrameDatasetSplit, SplitFullFrameDataset
from pathlib import Path

from bucketed_scene_flow_eval.datasets import construct_dataset
from bucketed_scene_flow_eval.datastructures import TimeSyncedSceneFlowFrame, EgoLidarFlow
from dataclasses import dataclass


@dataclass
class RawFullFrameInputSequence(BaseInputSequence):
    frame_list: list[TimeSyncedSceneFlowFrame]


@dataclass
class RawFullFrameOutputSequence(BaseOutputSequence):
    flow_list: list[EgoLidarFlow]


class RawFullFrameDataset(SplitFullFrameDataset):
    def __init__(
        self,
        dataset_name: str,
        root_dir: Path,
        set_length: int | None = None,
        split: FullFrameDatasetSplit | dict[str, int] = FullFrameDatasetSplit(0, 1),
        **kwargs,
    ):
        super().__init__(split=split, set_length=set_length)
        self.dataset = construct_dataset(dataset_name, dict(root_dir=root_dir, **kwargs))

    def evaluator(self) -> EvalWrapper:
        return EvalWrapper(self.dataset)

    def collate_fn(self, batch: list[RawFullFrameInputSequence]) -> list[RawFullFrameInputSequence]:
        return batch

    def __getitem__(self, split_idx) -> RawFullFrameInputSequence:
        global_idx = self._get_global_idx(split_idx)
        frame_list = self.dataset[global_idx]
        return RawFullFrameInputSequence(frame_list=frame_list)
