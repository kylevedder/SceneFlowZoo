import torch

from dataclasses import dataclass
from pathlib import Path


@dataclass
class OptimCheckpointStateDicts:
    model: dict[str, torch.Tensor] | None
    optimizer: dict[str, torch.Tensor] | None
    scheduler: dict[str, torch.Tensor] | None
    epoch: int

    def __post_init__(self):
        assert self.model is None or isinstance(
            self.model, dict
        ), f"Expected None or dict, but got {type(self.model)}"
        assert self.optimizer is None or isinstance(
            self.optimizer, dict
        ), f"Expected None or dict, but got {type(self.optimizer)}"
        assert self.scheduler is None or isinstance(
            self.scheduler, dict
        ), f"Expected None or dict, but got {type(self.scheduler)}"
        assert isinstance(self.epoch, int), f"Expected int, but got {type(self.epoch)}"

    @staticmethod
    def default() -> "OptimCheckpointStateDicts":
        return OptimCheckpointStateDicts(None, None, None, 0)

    @staticmethod
    def from_checkpoint(checkpoint: Path) -> "OptimCheckpointStateDicts":
        checkpoint = torch.load(checkpoint)
        assert "model" in checkpoint, f"Expected 'model' in checkpoint, but got {checkpoint.keys()}"
        assert (
            "optimizer" in checkpoint
        ), f"Expected 'optimizer' in checkpoint, but got {checkpoint.keys()}"
        assert (
            "scheduler" in checkpoint
        ), f"Expected 'scheduler' in checkpoint, but got {checkpoint.keys()}"
        assert "epoch" in checkpoint, f"Expected 'epoch' in checkpoint, but got {checkpoint.keys()}"

        return OptimCheckpointStateDicts(
            checkpoint["model"],
            checkpoint["optimizer"],
            checkpoint["scheduler"],
            checkpoint["epoch"],
        )

    def to_checkpoint(self, checkpoint: Path) -> None:
        torch.save(
            {
                "model": self.model,
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
                "epoch": self.epoch,
            },
            checkpoint,
        )
