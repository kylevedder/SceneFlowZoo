from .model_state_dicts import OptimCheckpointStateDicts

from pathlib import Path
from mmengine import Config
from bucketed_scene_flow_eval.utils import load_json
from bucketed_scene_flow_eval.interfaces import AbstractDataset
from dataloaders import TorchFullFrameInputSequence, BaseDataset
import tempfile
import dataloaders
from core_utils.checkpointing import setup_model
import torch
from models import BaseOptimizationModel
from models.whole_batch_optimization import WholeBatchOptimizationLoop
from dataclasses import dataclass


@dataclass
class SequenceInfo:
    sequence_id: str
    sequence_length: int


@dataclass
class OptimCheckpointModelLoader:

    root_config: Path
    checkpoint: Path
    sequence_info: SequenceInfo | None

    def __post_init__(self):
        assert self.root_config.exists(), f"Root config {self.root_config} does not exist"
        assert self.checkpoint.exists(), f"Checkpoint {self.checkpoint} does not exist"

    @staticmethod
    def from_checkpoint_dirs(
        root_config: Path, checkpoint_root: Path, sequence_id: str, sequence_id_to_length_file: Path
    ) -> "OptimCheckpointModelLoader":

        def _load_sizes(sequence_id_to_length: Path) -> dict[str, int]:
            data = load_json(sequence_id_to_length)
            assert len(data) > 0, f"No data found in {sequence_id_to_length}"
            return data

        def _load_checkpoint_path(checkpoint_root: Path, sequence_id: Path) -> Path:
            checkpoint_dir = checkpoint_root / f"job_{sequence_id}"
            checkpoints = sorted(checkpoint_dir.glob("*.pth"))
            if len(checkpoints) == 0:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

            # Try to grab the best checkpoint
            best_weights = checkpoint_dir / "best_weights.pth"
            if best_weights.exists():
                return best_weights

            if len(checkpoints) > 1:
                raise ValueError(
                    f"Multiple checkpoints found in {checkpoint_dir}. None are best_weights.pth"
                )
            return checkpoints[0]

        # Load the checkpoint
        full_checkpoint = _load_checkpoint_path(checkpoint_root, sequence_id)

        # Validate that sequence ID is in the sequence_id_to_length file
        sequence_id_to_length = _load_sizes(sequence_id_to_length_file)
        assert sequence_id in sequence_id_to_length, f"Sequence ID {sequence_id} not found"

        print(f"Loading checkpoint {full_checkpoint}")

        return OptimCheckpointModelLoader(
            root_config,
            full_checkpoint,
            SequenceInfo(
                sequence_id,
                sequence_id_to_length[sequence_id],
            ),
        )

    @staticmethod
    def from_checkpoint(root_config: Path, checkpoint: Path) -> "OptimCheckpointModelLoader":
        return OptimCheckpointModelLoader(root_config, checkpoint, None)

    def monkey_patch_model_weights(
        self,
        expected_model_weights: dict[str, torch.Tensor],
        found_model_weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        print("Expected model weights")
        for key in expected_model_weights:
            print(key)

        print("Found model weights")
        for key in found_model_weights:
            print(key)

        return {
            key.replace(
                "model.nn_layers._orig_mod.", "model.nn_layers._orig_mod.1._orig_mod."
            ): value
            for key, value in found_model_weights.items()
            if not key.startswith("model.encoder_plus_nn_layers")
        }

    def load_model(
        self,
    ) -> tuple[BaseOptimizationModel, TorchFullFrameInputSequence, AbstractDataset]:
        # Load the checkpoint
        model_state_dicts = OptimCheckpointStateDicts.from_checkpoint(self.checkpoint)

        config = self._make_custom_config()
        dataset_sequence, base_dataset = self._load_dataset_info(config)
        abstract_dataset: AbstractDataset = base_dataset.dataset

        model_wrapper = setup_model(config, base_dataset.evaluator(), None)
        model_loop: WholeBatchOptimizationLoop = model_wrapper.model
        assert isinstance(model_loop, WholeBatchOptimizationLoop)
        model: BaseOptimizationModel = model_loop._construct_model(dataset_sequence)
        assert isinstance(model, BaseOptimizationModel)

        try:
            model.load_state_dict(model_state_dicts.model)
        except RuntimeError as e:
            print("Unable to load model state dict, monkey patching...")
            monkey_patched_weights = self.monkey_patch_model_weights(
                model.state_dict(), model_state_dicts.model
            )
            model.load_state_dict(monkey_patched_weights)
        return model, dataset_sequence, abstract_dataset

    def _load_dataset_info(self, cfg: Config) -> tuple[TorchFullFrameInputSequence, BaseDataset]:
        dataset = dataloaders.construct_dataset(cfg.test_dataset.name, cfg.test_dataset.args)
        assert len(dataset) == 1, f"Expected dataset of length 1, got {len(dataset)}"
        return dataset[0].to("cuda"), dataset

    def _make_custom_config(
        self,
    ) -> Config:
        if self.sequence_info is None:
            return Config.fromfile(self.root_config)

        custom_cfg_content = f"""
_base_="{self.root_config.absolute()}"
test_dataset=dict(
    args=dict(
        log_subset=["{self.sequence_info.sequence_id}"],
        subsequence_length={self.sequence_info.sequence_length},
        use_cache=False,
    )
)
"""
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir)
            custom_cfg = path / f"{self.root_config.stem}_{self.sequence_info.sequence_id}.py"
            custom_cfg.write_text(custom_cfg_content)
            return Config.fromfile(custom_cfg)
