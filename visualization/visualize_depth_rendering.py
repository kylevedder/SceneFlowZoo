import argparse
import torch
from pathlib import Path


from models.whole_batch_optimization.checkpointing.model_loader import OptimCheckpointModelLoader
from models.mini_batch_optimization import GigachadOccFlowModel
from models import ForwardMode
from bucketed_scene_flow_eval.datastructures import EgoLidarDistance


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
    model, full_sequence = model_loader.load_model()
    model: GigachadOccFlowModel
    assert isinstance(
        model, GigachadOccFlowModel
    ), f"Expected GigachadOccFlowModel, got {type(model)}"

    with torch.inference_mode():
        with torch.no_grad():
            (output,) = model(ForwardMode.VAL, [full_sequence.detach().requires_grad_(False)], None)

    output_occ_list: list[EgoLidarDistance] = output.to_ego_lidar_distance_list()
    assert len(output_occ_list) + 1 == len(
        full_sequence
    ), f"Expected {len(full_sequence) - 1} outputs, got {len(output_occ_list)}"

    for idx, ego_lidar_distance in enumerate(output_occ_list):
        source_pc = full_sequence.get_full_ego_pc(idx)

    breakpoint()


if __name__ == "__main__":
    main()
