import argparse
from pathlib import Path


from models.whole_batch_optimization.checkpointing.model_loader import (
    OptimCheckpointModelLoader,
)


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

    model_loader = OptimCheckpointModelLoader.from_checkpoint_dirs(
        args.config, args.checkpoint_root, args.sequence_id, args.sequence_id_to_length
    )
    model = model_loader.load_model()
    breakpoint()


if __name__ == "__main__":
    main()
