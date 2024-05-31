import argparse
from dataloaders import RawFullFrameInputSequence, RawFullFrameDataset
from bucketed_scene_flow_eval.datastructures import PointCloudFrame
from bucketed_scene_flow_eval.utils import load_json, save_feather
from pathlib import Path
from models.tracker import CoTracker3D, BasePointTracker3D
import tqdm
import pandas as pd


def make_dataset(
    dataset_name: str,
    root_dir: Path,
    sequence_length: int,
    sliding_window: int,
    sequence_id: str | None,
    camera_names: list[str],
) -> RawFullFrameDataset:

    args_dict = dict(
        root_dir=root_dir,
        with_rgb=True,
        load_flow=False,
        subsequence_length=sequence_length,
        sliding_window_step_size=sliding_window,
        camera_names=camera_names,
        with_ground=False,
    )
    if sequence_id is not None:
        args_dict["log_subset"] = [sequence_id]

    return RawFullFrameDataset(
        dataset_name=dataset_name,
        **args_dict,
    )


def process_sequence(
    frame_sequence: RawFullFrameInputSequence, tracker: BasePointTracker3D
) -> list[PointCloudFrame]:
    pc_frame_list: list[PointCloudFrame] = tracker(frame_sequence)
    return pc_frame_list


def save_individual_track(
    frame_sequence: RawFullFrameInputSequence,
    pc_frame_list: list[PointCloudFrame],
    output_dir: Path,
    verbose: bool,
) -> None:
    window_starter_idx = frame_sequence.frame_list[0].log_idx
    save_dir = (
        output_dir
        / frame_sequence.frame_list[0].log_id
        / "sensors"
        / "camera_pc"
        / f"{window_starter_idx:06d}"
    )
    save_dir.mkdir(exist_ok=True, parents=True)
    for idx, pc_frame in enumerate(pc_frame_list):
        save_path = save_dir / f"{idx:06d}.feather"
        ego_pc = pc_frame.ego_pc.points
        # Convert to dataframe with columns named x, y, z
        ego_pc = pd.DataFrame(ego_pc, columns=["x", "y", "z"])
        save_feather(save_path, ego_pc, verbose=verbose)


def process_loop(
    dataset: RawFullFrameDataset, tracker: BasePointTracker3D, output_dir: Path, verbose: bool
):
    dataset_idxes = list(range(len(dataset)))
    for idx in tqdm.tqdm(dataset_idxes):
        frame_sequence = dataset[idx]
        track_pc_frames = process_sequence(frame_sequence, tracker)
        # This saves the point cloud from the individual track.
        save_individual_track(frame_sequence, track_pc_frames, output_dir, verbose)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--camera_config_file",
        type=Path,
        default=Path("data_prep_scripts/argo/av2_camera_tracking_config.json"),
    )
    parser.add_argument("--sequence_length", type=int, default=30)
    parser.add_argument("--sliding_window", type=int, default=10)

    parser.add_argument("--sequence_id", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dataset_name: str = args.dataset_name
    root_dir: Path = args.root_dir
    output_dir: Path = args.output_dir

    assert root_dir.is_dir(), f"Root directory {root_dir} does not exist"
    output_dir.mkdir(exist_ok=True, parents=True)

    camera_config_file: Path = args.camera_config_file
    assert camera_config_file.is_file(), f"Camera config file {camera_config_file} does not exist"

    camera_config: list[tuple[str, str]] = load_json(camera_config_file)

    sequence_length: int = args.sequence_length
    sliding_window: int = args.sliding_window
    sequence_id: str = args.sequence_id
    verbose: bool = args.verbose

    dataset = make_dataset(
        dataset_name,
        root_dir,
        sequence_length,
        sliding_window,
        sequence_id,
        [cam_name for cam_name, _ in camera_config],
    )
    tracker = CoTracker3D(camera_infos=camera_config).cuda()

    print(f"Starting processing loop with dataset of length {len(dataset)}...")
    process_loop(dataset, tracker, output_dir, verbose)


if __name__ == "__main__":
    main()
