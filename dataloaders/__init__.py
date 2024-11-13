from .abstract_dataset import BaseDataset, EvalWrapper, TorchEvalWrapper
from .dataclasses import (
    TorchFullFrameInputSequence,
    TorchFullFrameOutputSequence,
    FreeSpaceRays,
    TorchFullFrameOutputSequenceWithDistance,
)


# Defined before the importable classes to avoid circular imports if they use this function.
def construct_dataset(name: str, args: dict) -> BaseDataset:
    name = name.lower()
    if name not in name_to_class_lookup:
        raise ValueError(f"Unknown dataset name: {name}")

    cls = name_to_class_lookup[name]
    return cls(**args)


from .torch_full_frame_dataset import TorchFullFrameDataset
from .raw_full_frame_dataset import (
    RawFullFrameDataset,
    RawFullFrameInputSequence,
    RawFullFrameOutputSequence,
)


importable_classes = [TorchFullFrameDataset, RawFullFrameDataset]

name_to_class_lookup = {cls.__name__.lower(): cls for cls in importable_classes}


__all__ = [
    "TorchFullFrameDataset",
    "RawFullFrameDataset",
    "SequenceMinibatcher",
    "EvalWrapper",
    "TorchEvalWrapper",
    "TorchFullFrameInputSequence",
    "TorchFullFrameOutputSequence",
    "RawFullFrameInputSequence",
    "RawFullFrameOutputSequence",
    "construct_dataset",
    "MiniBatchedSceneFlowInputSequence",
]
