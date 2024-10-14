from .abstract_scene_flow_dataset import AbstractSceneFlowDataset, EvalWrapper
from .dataclasses import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence

# Defined before the importable classes to avoid circular imports if they use this function.
def construct_dataset(name: str, args: dict) -> AbstractSceneFlowDataset:
    name = name.lower()
    if name not in name_to_class_lookup:
        raise ValueError(f"Unknown dataset name: {name}")

    cls = name_to_class_lookup[name]
    return cls(**args)


from .scene_trajectory_benchmark_scene_flow_dataset import BucketedSceneFlowDataset
from .flow4ddataclasses import Flow4DSceneFlowDataset


importable_classes = [
    BucketedSceneFlowDataset,
    Flow4DSceneFlowDataset
]

name_to_class_lookup = {cls.__name__.lower(): cls for cls in importable_classes}


__all__ = [
    "BucketedSceneFlowDataset",
    "SequenceMinibatcher",
    "EvalWrapper",
    "BucketedSceneFlowInputSequence",
    "BucketedSceneFlowOutputSequence",
    "construct_dataset",
    "MiniBatchedSceneFlowInputSequence",
    "Flow4DSceneFlowDataset",
]
