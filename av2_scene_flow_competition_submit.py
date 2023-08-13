import os
# Set OMP env num threads to 1 to avoid deadlock in multithreading
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from loader_utils import load_npz
import multiprocessing as mp
import tqdm
from typing import Dict, Any, Tuple, List, Union
import open3d as o3d
from dataloaders import ArgoverseRawSequenceLoader, ArgoverseRawSequence
from pointclouds import PointCloud
import copy

# Parse arguments from command line for scene flow masks and scene flow outputs
parser = argparse.ArgumentParser()
parser.add_argument("scene_flow_mask_dir",
                    type=Path,
                    help="Path to scene flow mask directory")
parser.add_argument("scene_flow_output_dir",
                    type=Path,
                    help="Path to scene flow output directory")
parser.add_argument("argoverse_dir",
                    type=Path,
                    help="Path to scene flow output directory")
parser.add_argument("output_dir", type=Path, help="Path to output directory")
# Num CPUs to use for multiprocessing
parser.add_argument("--num_cpus",
                    type=int,
                    default=mp.cpu_count(),
                    help="Number of cpus to use for multiprocessing")
# bool flag for subset of data to use
parser.add_argument("--subset",
                    action="store_true",
                    help="Whether to use a subset of the data")
parser.add_argument("--visualize",
                    action="store_true",
                    help="Whether to visualize the data")
args = parser.parse_args()

assert args.scene_flow_mask_dir.is_dir(
), f"{args.scene_flow_mask_dir} is not a directory"
assert args.scene_flow_output_dir.is_dir(
), f"{args.scene_flow_output_dir} is not a directory"
assert args.argoverse_dir.is_dir(), f"{args.argoverse_dir} is not a directory"

args.output_dir.mkdir(parents=True, exist_ok=True)


def load_feather(filepath: Path):
    filepath = Path(filepath)
    assert filepath.exists(), f'{filepath} does not exist'
    return pd.read_feather(filepath)


def load_scene_flow_mask_from_folder(sequence_folder: Path) -> Dict[int, Any]:
    files = sorted(sequence_folder.glob("*.feather"))
    masks = [load_feather(file)['mask'].to_numpy() for file in files]
    return sequence_folder.stem, {
        int(file.stem): mask
        for file, mask in zip(files, masks)
    }


def load_scene_flow_predictions_from_folder(
        sequence_folder: Path) -> List[Any]:
    files = sorted(sequence_folder.glob("*.npz"))
    return sequence_folder.stem, [
        dict(load_npz(file, verbose=False)) for file in files
    ]


def multiprocess_load(folder: Path, worker_fn) -> Dict[str, Any]:
    sequence_folders = sorted(e for e in folder.glob("*") if e.is_dir())
    if args.subset:
        sequence_folders = sequence_folders[1:2]
    sequence_lookup = {}
    if args.num_cpus > 1:
        with mp.Pool(processes=args.num_cpus) as pool:
            for k, v in tqdm.tqdm(pool.imap_unordered(worker_fn,
                                                      sequence_folders),
                                  total=len(sequence_folders)):
                sequence_lookup[k] = v
    else:
        for sequence_folder in tqdm.tqdm(sequence_folders):
            k, v = worker_fn(sequence_folder)
            sequence_lookup[k] = v
    return sequence_lookup


print("Loading scene flow masks...")
sequence_mask_lookup = multiprocess_load(args.scene_flow_mask_dir,
                                         load_scene_flow_mask_from_folder)
print("Loading scene flow outputs...")
sequence_prediction_lookup = multiprocess_load(
    args.scene_flow_output_dir, load_scene_flow_predictions_from_folder)
print("Done loading scene flow masks and outputs")

mask_keys = set(sequence_mask_lookup.keys())
output_keys = set(sequence_prediction_lookup.keys())

if output_keys & mask_keys == mask_keys:
    print("All mask keys are in output keys")
    if len(output_keys - mask_keys) > 0:
        print("WARNING: extra output keys not in mask keys, removing them")
        sequence_prediction_lookup = {
            k: v
            for k, v in sequence_prediction_lookup.items() if k in mask_keys
        }
else:
    assert mask_keys == output_keys, f"Mask keys has {len(mask_keys - output_keys)} more keys than output keys, and output keys has {len(output_keys - mask_keys)} more keys than mask keys, and their intersection has {len(mask_keys & output_keys)} keys"

sequence_loader = ArgoverseRawSequenceLoader(args.argoverse_dir)


def make_lineset(pc, flowed_pc, draw_color):
    line_set = o3d.geometry.LineSet()
    assert len(pc) == len(
        flowed_pc
    ), f'pc and flowed_pc must have same length, but got {len(pc)} and {len(flowed_pc)}'
    line_set_points = np.concatenate([pc, flowed_pc], axis=0)

    lines = np.array([[i, i + len(pc)] for i in range(len(pc))])
    line_set.points = o3d.utility.Vector3dVector(line_set_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    draw_color = {
        'green': [0, 1, 0],
        'blue': [0, 0, 1],
        'red': [1, 0, 0],
        'yellow': [1, 1, 0],
        'orange': [1, 0.5, 0],
        'purple': [0.5, 0, 1],
        'pink': [1, 0, 1],
        'cyan': [0, 1, 1],
        'white': [1, 1, 1],
    }[draw_color]
    line_set.colors = o3d.utility.Vector3dVector(
        [draw_color for _ in range(len(lines))])
    return line_set


def merge_predictions_and_mask_data(sequence: ArgoverseRawSequence,
                                    timestamp_to_mask: Dict[int, Any],
                                    predictions: List[Any]) -> Dict[int, Any]:

    # Predictions are over frame pairs, so there are N - 1 predictions for N frames in a sequence.
    assert len(predictions) == len(
        sequence
    ) - 1, f"sequence id {sequence.log_id} len(id_to_output) {len(predictions)} != len(sequence) - 1 {len(sequence) - 1}"

    timestamp_to_sequence_frame_idx = {
        timestamp: idx
        for idx, timestamp in enumerate(sequence.timestamp_list)
    }

    timestamp_to_prediction = {
        t: v
        for t, v in zip(sequence.timestamp_list, predictions)
    }

    timestamp_to_masked_output = {}

    # Check that the masks are corresponding to the lidar files.
    for mask_timestamp in timestamp_to_mask:
        frame_idx = timestamp_to_sequence_frame_idx[mask_timestamp]
        frame_dict = sequence.load(frame_idx, frame_idx)
        next_frame_ego_dict = sequence.load(frame_idx + 1, frame_idx + 1)
        next_frame_curr_dict = sequence.load(frame_idx + 1, frame_idx)
        submission_mask_data = timestamp_to_mask[mask_timestamp]
        prediction_data = copy.deepcopy(
            timestamp_to_prediction[mask_timestamp])

        assert len(frame_dict['relative_pc_with_ground']) == len(
            submission_mask_data
        ), f"len(frame_dict['relative_pc_with_ground']) {len(frame_dict['relative_pc_with_ground'])} != len(submission_mask_data) {len(submission_mask_data)}"

        prediction_data['pc'] = frame_dict['relative_pc']
        prediction_data['pc_with_ground'] = frame_dict[
            'relative_pc_with_ground']
        prediction_data['pc_with_ground_is_ground_points'] = frame_dict[
            'is_ground_points']
        prediction_data[
            'pc_with_ground_is_submission_points'] = submission_mask_data
        prediction_data['next_pc'] = next_frame_ego_dict['relative_pc']
        prediction_data['next_pc_with_ground'] = next_frame_ego_dict[
            'relative_pc_with_ground']
        prediction_data[
            'next_pc_with_ground_is_ground_points'] = next_frame_ego_dict[
                'is_ground_points']
        prediction_data['relative_pose'] = next_frame_curr_dict[
            'relative_pose']

        timestamp_to_masked_output[mask_timestamp] = prediction_data

    return timestamp_to_masked_output


def save_sequence(sequence: ArgoverseRawSequence,
                  timestamp_to_mask: Dict[int, Any], predictions: List[Any]):
    sequence_name = sequence.log_id
    timestamp_to_masked_output = merge_predictions_and_mask_data(
        sequence, timestamp_to_mask, predictions)
    # print(f"Saving sequence {sequence_name}")
    for timestamp, data in sorted(timestamp_to_masked_output.items()):
        save_path = args.output_dir / sequence_name / f"{timestamp}.feather"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Sized to the full point cloud, including ground points.
        full_size_flow_array = np.zeros((len(data['pc_with_ground']), 3),
                                        dtype=data['flow'].dtype)
        no_ground_flow_array = np.zeros((len(data['pc']), 3),
                                        dtype=data['flow'].dtype)

        # Assign the flow to the non-ground and valid flow points.
        no_ground_flow_array[data['valid_idxes']] = data['flow']

        # Transform the flow from being in second frame to being in first frame.
        # The relative pose describes the transform from first to second frame,
        # so we need to rotate by the inverse of the relative pose.
        no_ground_flow_array = (data['relative_pose'].inverse().rotation_matrix
                                @ no_ground_flow_array.T).T

        full_size_flow_array[
            ~data['pc_with_ground_is_ground_points']] = no_ground_flow_array
        full_size_is_dynamic = (np.linalg.norm(full_size_flow_array, axis=1) >
                                0.05)

        # Add ego motion to the flow -- evaluation is performed on uncompenstated flow.
        # This is done by taking the point cloud and transforming it with the relative pose, and then computing the point differences.

        # Transform the point cloud with the relative pose from t to t+1.
        pc_with_ground_transformed = data['pc_with_ground'].transform(
            data['relative_pose'].inverse())
        point_diffs = pc_with_ground_transformed.matched_point_diffs(
            data['pc_with_ground'])
        # Add the point differences to the flow.
        full_size_flow_array += point_diffs

        # Extract the submission mask points from scratch array.
        output_flow_array = full_size_flow_array[
            data["pc_with_ground_is_submission_points"]]
        output_is_dynamic = full_size_is_dynamic[
            data["pc_with_ground_is_submission_points"]]

        output_pc = data['pc_with_ground'][
            data["pc_with_ground_is_submission_points"]]

        def resize_output_pc_next():
            output_pc_next = data['next_pc_with_ground'].points
            if len(output_pc_next) > len(output_pc):
                return output_pc_next[:len(output_pc)]
            else:
                output_scratch = np.zeros_like(output_pc)
                output_scratch[:len(output_pc_next)] = output_pc_next
                return output_scratch

        output_pc_next = resize_output_pc_next()

        df = pd.DataFrame(output_flow_array.astype(np.float16),
                          columns=['flow_tx_m', 'flow_ty_m', 'flow_tz_m'])
        df['is_dynamic'] = output_is_dynamic
        # Debug info not needed for final submission
        # df[['point_x', 'point_y', 'point_z']] = output_pc
        # df[['point_x_next', 'point_y_next', 'point_z_next']] = output_pc_next
        if save_path.exists():
            save_path.unlink()
        df.to_feather(save_path)

        if args.visualize:
            print("Sequence", sequence_name, "Timestamp", timestamp)

            def get_gt_flow():
                gt_path = Path("/efs/argoverse2/val_official_sceneflow/"
                               ) / sequence.log_id / f"{timestamp}.feather"
                assert gt_path.exists(), f"gt_path {gt_path} does not exist"
                gt_df = pd.read_feather(gt_path)
                gt_flow = gt_df[['flow_tx_m', 'flow_ty_m',
                                 'flow_tz_m']].to_numpy()
                return gt_flow

            # Visualize the lidar data and the flow data with Open3D
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(output_pc)
            pc.paint_uniform_color([0.1, 0, 0])

            next_pc = o3d.geometry.PointCloud()
            next_pc.points = o3d.utility.Vector3dVector(
                data['next_pc_with_ground'])
            next_pc.paint_uniform_color([0, 0, 1])

            pred_flow_lineset = make_lineset(output_pc,
                                             output_pc + output_flow_array,
                                             'red')
            gt_flow_lineset = make_lineset(output_pc,
                                           output_pc + get_gt_flow(), 'green')

            print("Num dynamic points", output_is_dynamic.sum(),
                  "Percent dynamic",
                  output_is_dynamic.sum() / len(output_is_dynamic))

            # visualize
            o3d.visualization.draw_geometries(
                [pc, next_pc, pred_flow_lineset, gt_flow_lineset])


def save_wrapper(args):
    save_sequence(*args)


print("Preparing args...")

save_args = [(sequence_loader.load_sequence(sequence_name),
              sequence_mask_lookup[sequence_name],
              sequence_prediction_lookup[sequence_name])
             for sequence_name in tqdm.tqdm(sorted(sequence_mask_lookup))]

print("Saving sequences...")

if args.num_cpus > 1:
    with mp.Pool(processes=args.num_cpus) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(save_wrapper, save_args),
                           total=len(save_args)):
            pass
else:
    for arg in tqdm.tqdm(save_args):
        save_wrapper(arg)
