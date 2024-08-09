from bucketed_scene_flow_eval.interfaces import (
    AbstractSequenceLoader,
    AbstractAVLidarSequence,
)
from bucketed_scene_flow_eval.datasets.argoverse2 import ArgoverseSceneFlowSequenceLoader
from bucketed_scene_flow_eval.datastructures import TimeSyncedSceneFlowFrame
from bucketed_scene_flow_eval.datasets import construct_dataset

from pathlib import Path
import argparse

from visualization.vis_lib import SequenceVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Path to the sequence folder
    parser.add_argument("--sequence_folder", type=Path, default="/efs/argoverse2/val/")
    parser.add_argument("--sequence_id", type=str, default="0b86f508-5df9-4a46-bc59-5b9536dbde9f")
    parser.add_argument("--sequence_length", type=int, default=150)
    # Path to the flow folder
    parser.add_argument(
        "--flow_folders",
        type=str,
        nargs="+",
        default=["/efs/argoverse2/val_sceneflow_feather/"],
        help="Path(s) to the flow folder(s). This can be the ground truth flow, or dumped results from a model. Multiple paths can be provided.",
    )
    parser.add_argument("--point_size", type=float, default=3)
    parser.add_argument("--frame_idx_step_size", type=int, default=1)
    parser.add_argument("--frame_idx_start", type=int, default=0)

    args = parser.parse_args()
    return args


def flow_folder_to_method_name(flow_folder: Path) -> str:

    relevant_folder = flow_folder

    skip_strings = ["sequence_len", "LoaderType"]
    while any(skip_string in relevant_folder.name for skip_string in skip_strings):
        relevant_folder = relevant_folder.parent

    return relevant_folder.name


def load_sequences(
    sequence_data_folder: Path, flow_folders: list[Path], sequence_id: str, sequence_length: int
) -> list[tuple[str, list[TimeSyncedSceneFlowFrame]]]:

    def load_sequence(flow_folder: Path) -> list[TimeSyncedSceneFlowFrame]:
        dataset = construct_dataset(
            name="Argoverse2NonCausalSceneFlow",
            args=dict(
                root_dir=sequence_data_folder,
                subsequence_length=sequence_length,
                with_ground=False,
                range_crop_type="ego",
                use_gt_flow=False,
                log_subset=[sequence_id],
                flow_data_path=flow_folder,
            ),
        )
        sequence = dataset[0]
        return sequence

    sequences = [load_sequence(flow_folder) for flow_folder in flow_folders]
    names = [flow_folder_to_method_name(flow_folder) for flow_folder in flow_folders]
    return list(zip(names, sequences))


def main():
    args = parse_args()

    sequence_folder = args.sequence_folder
    sequence_id = args.sequence_id
    sequence_length = args.sequence_length
    flow_folders = [Path(flow_folder) for flow_folder in args.flow_folders]
    point_size = args.point_size
    frame_idx_step_size = args.frame_idx_step_size
    frame_idx_start = args.frame_idx_start
    print("Sequence ID: ", sequence_id)

    for flow_folder in flow_folders:
        assert flow_folder.exists(), f"Flow folder {flow_folder} does not exist."
    sequences = load_sequences(sequence_folder, flow_folders, sequence_id, sequence_length)

    vis = SequenceVisualizer(
        sequences,
        sequence_id,
        subsequence_length=sequence_length,
        point_size=point_size,
        step_size=frame_idx_step_size,
        frame_idx=frame_idx_start,
    )
    vis.run()


if __name__ == "__main__":
    main()
