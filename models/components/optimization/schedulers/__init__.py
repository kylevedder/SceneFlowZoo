import torch
from dataclasses import dataclass
from torch.optim.lr_scheduler import LRScheduler
from .stopping_schedulers import (
    StoppingScheduler,
    ReduceLROnPlateauWithFloorRestart,
    PassThroughScheduler,
)
from core_utils.model_saver import ModelStateDicts


@dataclass
class SchedulerBuilder:
    name: str
    args: dict[str, object]

    def __post_init__(self):
        assert isinstance(self.name, str), "name must be a string"
        assert isinstance(self.args, dict), "args must be a dictionary"

    def to_scheduler(
        self, optimizer: torch.optim.Optimizer, model_state_dict: ModelStateDicts
    ) -> StoppingScheduler:
        return construct_scheduler(optimizer, self.name, self.args, model_state_dict)


def construct_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str,
    args: dict[str, object],
    model_state_dict: ModelStateDicts,
) -> StoppingScheduler:
    name = name.lower()
    if name not in name_to_class_lookup:
        raise ValueError(f"Unknown model name: {name}")

    cls = name_to_class_lookup[name]
    scheduler: LRScheduler = cls(optimizer=optimizer, **args)
    if model_state_dict.scheduler is not None:
        scheduler.load_state_dict(model_state_dict.scheduler)
    return scheduler


importable_schedulers = [
    StoppingScheduler,
    PassThroughScheduler,
    ReduceLROnPlateauWithFloorRestart,
]

# Ensure all importable models are based on the BaseModel class.
for cls in importable_schedulers:
    assert issubclass(cls, StoppingScheduler)
name_to_class_lookup = {cls.__name__.lower(): cls for cls in importable_schedulers}
