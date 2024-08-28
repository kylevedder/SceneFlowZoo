from bucketed_scene_flow_eval.datastructures import TimeSyncedSceneFlowFrame
from bucketed_scene_flow_eval.datasets import construct_dataset

from pathlib import Path
import argparse

from visualization.vis_lib import SequenceVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Path to the sequence folder
    parser.add_argument("sequence_folder", type=Path)
    # Path to the flow folder
    parser.add_argument(
        "--flow_folders",
        type=str,
        nargs="+",
        default=["/efs/argoverse2/val_sceneflow_feather/"],
        help="Path(s) to the flow folder(s). This can be the ground truth flow, or dumped results from a model. Multiple paths can be provided.",
    )
    parser.add_argument("--sequence_length", type=int, default=20)
    parser.add_argument("--point_size", type=float, default=3)
    parser.add_argument("--frame_idx_step_size", type=int, default=1)
    parser.add_argument("--frame_idx_start", type=int, default=0)
    parser.add_argument("--camera_pose_file", type=Path, default=None)

    args = parser.parse_args()
    return args


def flow_folder_to_method_name(flow_folder: Path | None) -> str:
    if flow_folder is None:
        return "None"

    relevant_folder = flow_folder

    skip_strings = ["sequence_len", "LoaderType"]
    while any(skip_string in relevant_folder.name for skip_string in skip_strings):
        relevant_folder = relevant_folder.parent

    return relevant_folder.name


def load_sequences(
    sequence_data_folder: Path, subsequence_legth: int, flow_folders: list[Path]
) -> list[tuple[str, list[TimeSyncedSceneFlowFrame]]]:

    def load_sequence(flow_folder: Path) -> list[TimeSyncedSceneFlowFrame]:
        dataset = construct_dataset(
            name="OrbbecAstra",
            args=dict(
                root_dir=sequence_data_folder,
                flow_dir=flow_folder,
                subsequence_length=subsequence_legth,
            ),
        )
        assert len(dataset) > 0, f"No sequences found in {sequence_data_folder}"
        sequence = dataset[0]
        return sequence

    sequences = [load_sequence(flow_folder) for flow_folder in flow_folders]
    names = [flow_folder_to_method_name(flow_folder) for flow_folder in flow_folders]
    return list(zip(names, sequences))


def main():
    args = parse_args()

    sequence_folder = args.sequence_folder
    flow_folders = [
        Path(flow_folder) if flow_folder.strip().lower() != "none" else None
        for flow_folder in args.flow_folders
    ]
    sequence_length = args.sequence_length
    point_size = args.point_size
    camera_pose_file = args.camera_pose_file
    sequence_id = sequence_folder.name
    frame_idx_start = args.frame_idx_start
    print("Sequence ID: ", sequence_id)

    for flow_folder in flow_folders:
        if flow_folder is not None:
            assert flow_folder.exists(), f"Flow folder {flow_folder} does not exist."
    sequences = load_sequences(sequence_folder, sequence_length, flow_folders)

    vis = SequenceVisualizer(
        sequences,
        sequence_id,
        subsequence_lengths=[sequence_length],
        point_size=point_size,
        frame_idx=frame_idx_start,
        color_map_name="zebra",
        add_world_frame=True,
    )
    vis.run(camera_pose_path=camera_pose_file)


if __name__ == "__main__":
    main()
