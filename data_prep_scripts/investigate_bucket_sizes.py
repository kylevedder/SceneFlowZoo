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

DYNAMIC_THRESHOLD = 0.5


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
        flowed_pc = supervised_frame['relative_flowed_pc'][valid_indices]
        flow = flowed_pc - pc
        
        flow_speed_meters_per_second = np.linalg.norm(flow,
                                                            axis=1) * 10.0
        
        if args.dataset == 'waymo':
            BACKGROUND_ID = 0
        else:
            BACKGROUND_ID = -1

        background_count = np.sum(classes == BACKGROUND_ID)
        foreground_static_count = np.sum(
            (classes > BACKGROUND_ID) & (flow_speed_meters_per_second <= DYNAMIC_THRESHOLD))
        foreground_dynamic_count = np.sum(
            (classes > BACKGROUND_ID) & (flow_speed_meters_per_second > DYNAMIC_THRESHOLD))

        num_points = len(pc)
        if num_points < 100:
            print("Seq idx", seq_idx, "idx", idx, "num_points", num_points)
        infos_lst.append({
            "background_count": background_count,
            "foreground_static_count": foreground_static_count,
            "foreground_dynamic_count": foreground_dynamic_count,
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
        for idx, seq_id in enumerate(tqdm.tqdm(seq_ids)))
    pc_infos_lst = [item for sublist in pc_infos_lst for item in sublist]
else:
    pc_infos_lst = []
    for idx, seq_id in enumerate(tqdm.tqdm(seq_ids)):
        supervised_seq = supervised_seq_loader.load_sequence(seq_id)
        unsupervised_seq = unsupervised_seq_loader.load_sequence(seq_id)
        pc_infos = get_pc_infos(idx, supervised_seq, unsupervised_seq)
        pc_infos_lst.extend(pc_infos)

save_pickle(
    f"validation_results/{args.dataset}_validation_pointcloud_pc_buckets.pkl",
    pc_infos_lst)

background_counts = [e["background_count"] for e in pc_infos_lst]
foreground_static_counts = [e["foreground_static_count"] for e in pc_infos_lst]
foreground_dynamic_counts = [e["foreground_dynamic_count"] for e in pc_infos_lst]


print("Background Count Sum:", np.sum(background_counts), "Mean:", np.mean(background_counts), "Std:", np.std(background_counts), "Median:", np.median(background_counts))
print("Foreground Static Count Sum:", np.sum(foreground_static_counts), "Mean:", np.mean(foreground_static_counts), "Std:", np.std(foreground_static_counts), "Median:", np.median(foreground_static_counts))
print("Foreground Dynamic Count Sum:", np.sum(foreground_dynamic_counts), "Mean:", np.mean(foreground_dynamic_counts), "Std:", np.std(foreground_dynamic_counts), "Median:", np.median(foreground_dynamic_counts))