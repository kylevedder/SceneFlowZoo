from .model_state_dicts import OptimCheckpointStateDicts

from pathlib import Path
from mmengine import Config
from bucketed_scene_flow_eval.utils import load_json
from dataloaders import TorchFullFrameInputSequence, BaseDataset
import tempfile
import dataloaders
from core_utils.checkpointing import setup_model


class OptimCheckpointModelLoader:

    def __init__(
        self, root_config: Path, checkpoint: Path, sequence_id: str, sequence_length: int
    ) -> None:
        self.root_config = root_config
        self.checkpoint = checkpoint
        self.sequence_id = sequence_id
        self.sequence_length = sequence_length

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
            assert len(checkpoints) == 1, f"No checkpoints found in {checkpoint_dir}"
            return checkpoints[0]

        # Load the checkpoint
        full_checkpoint = _load_checkpoint_path(checkpoint_root, sequence_id)

        # Validate that sequence ID is in the sequence_id_to_length file
        sequence_id_to_length = _load_sizes(sequence_id_to_length_file)
        assert sequence_id in sequence_id_to_length, f"Sequence ID {sequence_id} not found"

        return OptimCheckpointModelLoader(
            root_config,
            full_checkpoint,
            sequence_id,
            sequence_id_to_length[sequence_id],
        )

    def load_model(self):
        # Load the checkpoint
        model_state_dicts = OptimCheckpointStateDicts.from_checkpoint(self.checkpoint)

        config = self._make_custom_config()
        dataset_sequence, base_dataset = self._load_dataset_info(config)

        model_wrapper = setup_model(config, base_dataset.evaluator(), None)
        model_loop = model_wrapper.model
        model = model_loop._construct_model(dataset_sequence)

        model.load_state_dict(model_state_dicts.model)
        return model

    def _load_dataset_info(self, cfg: Config) -> tuple[TorchFullFrameInputSequence, BaseDataset]:
        dataset = dataloaders.construct_dataset(cfg.test_dataset.name, cfg.test_dataset.args)
        assert len(dataset) == 1, f"Expected dataset of length 1, got {len(dataset)}"
        return dataset[0].to("cuda"), dataset

    def _make_custom_config(
        self,
    ) -> Config:
        custom_cfg_content = f"""
_base_="{self.root_config.absolute()}"
test_dataset=dict(
    args=dict(
        log_subset=["{self.sequence_id}"],
        subsequence_length={self.sequence_length},
        use_cache=False,
    )
)
"""
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir)
            custom_cfg = path / f"{self.root_config.stem}_{self.sequence_id}.py"
            custom_cfg.write_text(custom_cfg_content)
            return Config.fromfile(custom_cfg)
