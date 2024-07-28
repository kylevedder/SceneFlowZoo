from core_utils import setup_model
import argparse
from pathlib import Path
from mmengine import Config
from bucketed_scene_flow_eval.utils import load_json
from dataloaders import TorchFullFrameOutputSequence
from models.whole_batch_optimization import WholeBatchOptimizationLoop
from bucketed_scene_flow_eval.datastructures import EgoLidarFlow
from models import ForwardMode
import tempfile
import torch
import dataloaders


def _load_sizes(sequence_id_to_length: Path) -> dict[str, int]:
    data = load_json(sequence_id_to_length)
    assert len(data) > 0, f"No data found in {sequence_id_to_length}"
    return data


def _load_checkpoint_path(checkpoint_root: Path, sequence_id: Path) -> Path:
    checkpoint_dir = checkpoint_root / f"job_{sequence_id}"
    checkpoints = sorted(checkpoint_dir.glob("*.pth"))
    assert len(checkpoints) == 1, f"No checkpoints found in {checkpoint_dir}"
    return checkpoints[0]


def _make_custom_config(base_cfg: Path, sequence_id: Path, sequence_length: int) -> Config:
    custom_cfg_content = f"""
_base_="{base_cfg.absolute()}"
test_dataset=dict(
    args=dict(
        log_subset=["{sequence_id}"],
        subsequence_length={sequence_length},
    )
)
"""
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir)
        custom_cfg = path / f"{base_cfg.stem}_{sequence_id}.py"
        custom_cfg.write_text(custom_cfg_content)
        return Config.fromfile(custom_cfg)


def get_frame_list(cfg: torch.utils.data.dataloader.DataLoader):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("sequence_id", type=str)
    parser.add_argument(
        "--sequence_id_to_length",
        type=Path,
        default=Path("data_prep_scripts/argo/av2_test_sizes.json"),
    )
    args = parser.parse_args()

    sequence_id_to_length = _load_sizes(args.sequence_id_to_length)
    assert args.sequence_id in sequence_id_to_length, f"Sequence ID {args.sequence_id} not found"
    checkpoint = _load_checkpoint_path(args.checkpoint_root, args.sequence_id)

    cfg = _make_custom_config(
        args.config, args.sequence_id, sequence_id_to_length[args.sequence_id]
    )

    dataset = dataloaders.construct_dataset(cfg.test_dataset.name, cfg.test_dataset.args)
    assert len(dataset) == 1, f"Expected dataset of length 1, got {len(dataset)}"
    dataset_entry = dataset[0].to("cuda")

    model_wrapper = setup_model(cfg, dataset.evaluator(), None)
    model: WholeBatchOptimizationLoop = model_wrapper.model
    assert isinstance(
        model, WholeBatchOptimizationLoop
    ), f"Expected WholeBatchOptimizationLoop, got {type(model)}"
    model.checkpoint = checkpoint
    model.eval_only = True
    (results,) = model(ForwardMode.VAL, [dataset_entry], model_wrapper.logger)
    results: TorchFullFrameOutputSequence
    ego_flows: list[EgoLidarFlow] = results.to_ego_lidar_flow_list()
    breakpoint()


if __name__ == "__main__":
    main()
