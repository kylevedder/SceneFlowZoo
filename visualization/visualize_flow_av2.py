from bucketed_scene_flow_eval.interfaces import (
    AbstractSequenceLoader,
    AbstractAVLidarSequence,
)
from bucketed_scene_flow_eval.datasets.argoverse2 import ArgoverseSceneFlowSequenceLoader

from pathlib import Path
import argparse

from visualization.vis_lib import SequenceVisualizer

parser = argparse.ArgumentParser()
parser.add_argument("--sequence_id", type=str, default="0b86f508-5df9-4a46-bc59-5b9536dbde9f")
parser.add_argument("--sequence_start_idx", type=int, default=125)
# Path to the sequence folder
parser.add_argument("--sequence_folder", type=Path, default="/efs/argoverse2/val/")
# Path to the flow folder
parser.add_argument(
    "--flow_folders",
    type=str,
    nargs="+",  # This allows the user to input multiple flow folder paths
    default=["/efs/argoverse2/val_sceneflow_feather/"],
    help="Path(s) to the flow folder(s). This can be the ground truth flow, or dumped results from a model. Multiple paths can be provided.",
)
parser.add_argument("--point_size", type=float, default=3)
parser.add_argument("--frame_idx_step_size", type=int, default=1)
parser.add_argument("--sequence_length", type=int, default=2)
args = parser.parse_args()

sequence_id = args.sequence_id
print("Sequence ID: ", sequence_id)

flow_folders = [Path(flow_folder) for flow_folder in args.flow_folders]
for flow_folder in flow_folders:
    assert flow_folder.exists(), f"Flow folder {flow_folder} does not exist."


def flow_folder_to_method_name(flow_folder: Path) -> str:

    relevant_folder = flow_folder

    skip_strings = ["sequence_len", "LoaderType"]
    while any(skip_string in relevant_folder.name for skip_string in skip_strings):
        relevant_folder = relevant_folder.parent

    return relevant_folder.name


def load_sequences(
    sequence_data_folder: Path, flow_folders: list[Path], sequence_id: str
) -> list[tuple[str, AbstractAVLidarSequence]]:
    sequence_loaders: list[AbstractSequenceLoader] = [
        ArgoverseSceneFlowSequenceLoader(
            sequence_data_folder,
            flow_folder,
            use_gt_flow=False,
        )
        for flow_folder in flow_folders
    ]

    sequences = [sequence_loader.load_sequence(sequence_id) for sequence_loader in sequence_loaders]
    names = [flow_folder_to_method_name(flow_folder) for flow_folder in flow_folders]

    sequence_and_names = list(zip(names, sequences))

    sequence_lengths = [len(sequence) for sequence in sequences]
    if not all(length == sequence_lengths[0] for length in sequence_lengths):
        length_strs = "\n".join(
            [f"{name}:\t{length}" for name, length in zip(names, sequence_lengths)]
        )
        # Warning:
        print(f"Warning: Not all sequences have the same length. Lengths:\n{length_strs}")

    return sequence_and_names


sequences = load_sequences(args.sequence_folder, flow_folders, sequence_id)

vis = SequenceVisualizer(
    sequences,
    sequence_id,
    frame_idx=args.sequence_start_idx,
    subsequence_length=args.sequence_length,
    point_size=args.point_size,
    step_size=args.frame_idx_step_size,
)
vis.run()
