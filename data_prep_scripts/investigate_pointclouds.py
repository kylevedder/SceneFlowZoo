import os
# set OMP_NUM_THREADS before importing torch or numpy
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
from pathlib import Path
import numpy as np
import tqdm
import joblib
import multiprocessing
from core_utils import save_pickle
import open3d as o3d
from pointclouds import PointCloud
import torch

from dataloaders import ArgoverseSupervisedFlowSequenceLoader, ArgoverseUnsupervisedFlowSequenceLoader, WaymoSupervisedFlowSequenceLoader, WaymoUnsupervisedFlowSequenceLoader

# cli argument to pick argo or waymo
parser = argparse.ArgumentParser()
parser.add_argument('dataset',
                    type=str,
                    help='argo or waymo',
                    choices=['argo', 'waymo'])
parser.add_argument('--cpus',
                    type=int,
                    help='number of cpus to use',
                    default=multiprocessing.cpu_count() - 1)
args = parser.parse_args()

argoverse_data = Path("/efs/argoverse2")
waymo_data = Path("/efs/waymo_open_processed_flow")

SLOW_THRESHOLD = 0.4
FAST_THRESHOLD = 1.0


def get_pc_infos(seq_idx, supervised_seq, unsupervised_seq):
    infos_lst = []
    max_range = min(len(supervised_seq), len(unsupervised_seq))
    for idx in range(max_range):
        if idx == max_range - 1:
            continue
        supervised_frame = supervised_seq.load(idx, idx)
        unsupervised_frame = unsupervised_seq.load(idx, idx)

        assert (
            supervised_frame['relative_pc'] ==
            unsupervised_frame['relative_pc']
        ), f"supervised_frame['relative_pc'] {supervised_frame['relative_pc']} != unsupervised_frame['relative_pc'] {unsupervised_frame['relative_pc']}"

        valid_indices = unsupervised_frame['valid_idxes']
        classes = supervised_frame['pc_classes'][valid_indices]
        pc = supervised_frame['relative_pc'][valid_indices]
        pseudoflow = unsupervised_frame['flow']
        if pseudoflow.ndim == 3 and pseudoflow.shape[0] == 1:
            pseudoflow = pseudoflow.squeeze(0)

        assert len(valid_indices) == len(
            pseudoflow
        ), f"len(valid_indices) {len(valid_indices)} != len(pseudoflow) {len(pseudoflow)}"

        pseudoflow_speed_meters_per_second = np.linalg.norm(pseudoflow,
                                                            axis=1) * 10.0

        assert len(pc) == len(
            pseudoflow_speed_meters_per_second
        ), f"len(pc) {len(pc)} != len(pseudoflow_speed) {len(pseudoflow_speed_meters_per_second)}"

        slow_point_count = np.sum(
            pseudoflow_speed_meters_per_second <= SLOW_THRESHOLD)
        medium_point_count = np.sum(
            (pseudoflow_speed_meters_per_second > SLOW_THRESHOLD)
            & (pseudoflow_speed_meters_per_second < FAST_THRESHOLD))
        fast_point_count = np.sum(
            pseudoflow_speed_meters_per_second >= FAST_THRESHOLD)

        foreground_count = np.sum(classes >= 0)
        background_count = np.sum(classes == -1)
        assert foreground_count + background_count == len(
            pc
        ), f"foreground_count {foreground_count} + background_count {background_count} != len(pc) {len(pc)}"

        num_points = len(pc)
        if num_points < 100:
            print("Seq idx", seq_idx, "idx", idx, "num_points", num_points)
        infos_lst.append({
            "num_points": num_points,
            "foreground_count": foreground_count,
            "background_count": background_count,
            "slow_point_count": slow_point_count,
            "medium_point_count": medium_point_count,
            "fast_point_count": fast_point_count,
        })
    print("Finished sequence",
          seq_idx,
          "with",
          len(supervised_seq),
          "frames",
          flush=True)
    return infos_lst


if args.dataset == 'waymo':
    supervised_seq_loader = WaymoSupervisedFlowSequenceLoader(waymo_data /
                                                              "training")
    unsupervised_seq_loader = WaymoUnsupervisedFlowSequenceLoader(
        waymo_data / "training", waymo_data / "train_nsfp_flow")
else:
    supervised_seq_loader = ArgoverseSupervisedFlowSequenceLoader(
        argoverse_data / "val", argoverse_data / "val_sceneflow")
    unsupervised_seq_loader = ArgoverseUnsupervisedFlowSequenceLoader(
        argoverse_data / "val", argoverse_data / "val_nsfp_flow")
seq_ids = supervised_seq_loader.get_sequence_ids()

if args.cpus > 1:
    # Use joblib to parallelize the loading of the sequences
    pc_infos_lst = joblib.Parallel(n_jobs=args.cpus)(
        joblib.delayed(get_pc_infos)(
            idx, supervised_seq_loader.load_sequence(seq_id),
            unsupervised_seq_loader.load_sequence(seq_id))
        for idx, seq_id in enumerate(seq_ids))
    pc_infos_lst = [item for sublist in pc_infos_lst for item in sublist]
else:
    pc_infos_lst = []
    for idx, seq_id in enumerate(tqdm.tqdm(seq_ids)):
        supervised_seq = supervised_seq_loader.load_sequence(seq_id)
        unsupervised_seq = unsupervised_seq_loader.load_sequence(seq_id)
        pc_infos = get_pc_infos(idx, supervised_seq, unsupervised_seq)
        pc_infos_lst.extend(pc_infos)

save_pickle(
    f"validation_results/{args.dataset}_validation_pointcloud_pc_infos.pkl",
    pc_infos_lst)

pc_sizes_lst = [e["num_points"] for e in pc_infos_lst]
foreground_counts = [e["foreground_count"] for e in pc_infos_lst]
background_counts = [e["background_count"] for e in pc_infos_lst]

slow_counts = [e["slow_point_count"] for e in pc_infos_lst]
medium_counts = [e["medium_point_count"] for e in pc_infos_lst]
fast_counts = [e["fast_point_count"] for e in pc_infos_lst]

print("PC Point Count Mean:", np.mean(pc_sizes_lst), "Std:",
      np.std(pc_sizes_lst), "Median:", np.median(pc_sizes_lst))
print("Foreground Mean:", np.mean(foreground_counts), "Std:",
      np.std(foreground_counts), "Median:", np.median(foreground_counts))
print("Background Mean:", np.mean(background_counts), "Std:",
      np.std(background_counts), "Median:", np.median(background_counts))
print("Slow Mean:", np.mean(slow_counts), "Std:", np.std(slow_counts),
      "Median:", np.median(slow_counts))
print("Medium Mean:", np.mean(medium_counts), "Std:", np.std(medium_counts),
      "Median:", np.median(medium_counts))
print("Fast Mean:", np.mean(fast_counts), "Std:", np.std(fast_counts),
      "Median:", np.median(fast_counts))
