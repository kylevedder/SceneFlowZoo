from typing import Any

from .base_models import (
    BaseTorchModel,
    BaseRawModel,
    ForwardMode,
    BaseOptimizationModel,
    AbstractBatcher,
)


def construct_model(name: str, args: dict[str, Any]) -> BaseTorchModel:
    name = name.lower()
    if name not in name_to_class_lookup:
        raise ValueError(f"Unknown model name: {name}")

    cls = name_to_class_lookup[name]
    return cls(**args)


from .constant_vector_baseline import ConstantVectorBaseline
from .feed_forward.deflow import DeFlow
from .feed_forward.fast_flow_3d import (
    FastFlow3D,
    FastFlow3DBucketedLoaderLoss,
    FastFlow3DSelfSupervisedLoss,
)
from .whole_batch_optimization import (
    NSFPForwardOnlyOptimizationLoop,
    NSFPCycleConsistencyOptimizationLoop,
    FastNSFModelOptimizationLoop,
    Liu2024OptimizationLoop,
)
from .mini_batch_optimization import (
    GigachadNSFOptimizationLoop,
    NTPOptimizationLoop,
    GigachadOccFlowOptimizationLoop,
)


importable_models = [
    DeFlow,
    FastFlow3D,
    ConstantVectorBaseline,
    NSFPForwardOnlyOptimizationLoop,
    NSFPCycleConsistencyOptimizationLoop,
    FastNSFModelOptimizationLoop,
    Liu2024OptimizationLoop,
    GigachadNSFOptimizationLoop,
    NTPOptimizationLoop,
    GigachadOccFlowOptimizationLoop,
]

# Ensure all importable models are based on the BaseModel class.
for cls in importable_models:
    assert issubclass(cls, BaseTorchModel) or issubclass(
        cls, BaseRawModel
    ), f"{cls} is not a valid model class."

name_to_class_lookup = {cls.__name__.lower(): cls for cls in importable_models}


__all__ = [
    "DeFlow",
    "FastFlow3D",
    "FastFlow3DBucketedLoaderLoss",
    "FastFlow3DSelfSupervisedLoss",
    "NSFPModel",
    "FastNSFModel",
    "FastNSFPlusPlusModel",
    "Liu2024Model",
    "GigaChadNSFModel",
    "ConstantVectorBaseline",
    "BaseTorchModel",
    "ForwardMode",
]
