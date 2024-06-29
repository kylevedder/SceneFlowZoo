import torch
from dataclasses import dataclass

from .stopping_schedulers import (
    StoppingScheduler,
    ReduceLROnPlateauWithFloorRestart,
    PassThroughScheduler,
)


@dataclass
class SchedulerBuilder:
    name: str
    args: dict[str, object]

    def __post_init__(self):
        assert isinstance(self.name, str), "name must be a string"
        assert isinstance(self.args, dict), "args must be a dictionary"

    def to_scheduler(self, optimizer: torch.optim.Optimizer) -> StoppingScheduler:
        return construct_scheduler(optimizer, self.name, self.args)


def construct_scheduler(
    optimizer: torch.optim.Optimizer, name: str, args: dict[str, object]
) -> StoppingScheduler:
    name = name.lower()
    if name not in name_to_class_lookup:
        raise ValueError(f"Unknown model name: {name}")

    cls = name_to_class_lookup[name]
    return cls(optimizer=optimizer, **args)


importable_schedulers = [
    StoppingScheduler,
    PassThroughScheduler,
    ReduceLROnPlateauWithFloorRestart,
]

# Ensure all importable models are based on the BaseModel class.
for cls in importable_schedulers:
    assert issubclass(cls, StoppingScheduler)
name_to_class_lookup = {cls.__name__.lower(): cls for cls in importable_schedulers}
