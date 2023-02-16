import torch
import pandas as pd
import open3d as o3d
from dataloaders import ArgoverseSupervisedFlowSequenceLoader, ArgoverseSupervisedFlowSequence
from pointclouds import PointCloud, SE3
import numpy as np
import tqdm
import joblib
import multiprocessing
from loader_utils import save_pickle

CATEGORY_NAME_TO_ID = {
    "ANIMAL": 0,
    "ARTICULATED_BUS": 1,
    "BICYCLE": 2,
    "BICYCLIST": 3,
    "BOLLARD": 4,
    "BOX_TRUCK": 5,
    "BUS": 6,
    "CONSTRUCTION_BARREL": 7,
    "CONSTRUCTION_CONE": 8,
    "DOG": 9,
    "LARGE_VEHICLE": 10,
    "MESSAGE_BOARD_TRAILER": 11,
    "MOBILE_PEDESTRIAN_CROSSING_SIGN": 12,
    "MOTORCYCLE": 13,
    "MOTORCYCLIST": 14,
    "OFFICIAL_SIGNALER": 15,
    "PEDESTRIAN": 16,
    "RAILED_VEHICLE": 17,
    "REGULAR_VEHICLE": 18,
    "SCHOOL_BUS": 19,
    "SIGN": 20,
    "STOP_SIGN": 21,
    "STROLLER": 22,
    "TRAFFIC_LIGHT_TRAILER": 23,
    "TRUCK": 24,
    "TRUCK_CAB": 25,
    "VEHICULAR_TRAILER": 26,
    "WHEELCHAIR": 27,
    "WHEELED_DEVICE": 28,
    "WHEELED_RIDER": 29,
    "NONE": -1
}

CATEGORY_ID_TO_NAME = {v: k for k, v in CATEGORY_NAME_TO_ID.items()}

CATEGORY_ID_TO_IDX = {
    v: idx
    for idx, v in enumerate(sorted(CATEGORY_NAME_TO_ID.values()))
}
CATEGORY_IDX_TO_ID = {v: k for k, v in CATEGORY_ID_TO_IDX.items()}

speed_bucket_ticks = [
    0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5,
    1.6, 1.7, 1.8, 1.9, 2.0, np.inf
]


def get_speed_bucket_ranges():
    return list(zip(speed_bucket_ticks, speed_bucket_ticks[1:]))


def process_sequence(sequence: ArgoverseSupervisedFlowSequence):

    speed_bucket_ranges = get_speed_bucket_ranges()
    count_array = np.zeros((len(CATEGORY_ID_TO_IDX), len(speed_bucket_ranges)),
                           dtype=np.int64)

    for idx in range(len(sequence)):
        frame = sequence.load(idx, 0)
        pc = frame['relative_pc']
        flowed_pc = frame['relative_flowed_pc']
        if flowed_pc is None:
            continue
        category_ids = frame['pc_classes']
        flow = flowed_pc.points - pc.points
        for category_id in np.unique(category_ids):
            category_idx = CATEGORY_ID_TO_IDX[category_id]
            category_mask = (category_ids == category_id)
            category_flow = flow[category_mask]
            # convert to m/s
            category_speeds = np.linalg.norm(category_flow, axis=1) * 10.0
            category_speed_buckets = np.digitize(category_speeds,
                                                 speed_bucket_ticks) - 1
            for speed_bucket_idx in range(len(speed_bucket_ranges)):
                bucket_mask = (category_speed_buckets == speed_bucket_idx)
                num_points = np.sum(bucket_mask)
                count_array[category_idx, speed_bucket_idx] += num_points
    return count_array


sequence_loader = ArgoverseSupervisedFlowSequenceLoader(
    '/efs/argoverse2/val/', '/efs/argoverse2/val_sceneflow/')

sequences = [
    sequence_loader.load_sequence(sequence_id)
    for sequence_id in sequence_loader.get_sequence_ids()
]

# counts_array_lst = [
#     process_sequence(sequence) for sequence in tqdm.tqdm(sequences)
# ]

# process sequences in parallel with joblib
counts_array_lst = joblib.Parallel(n_jobs=multiprocessing.cpu_count())(
    joblib.delayed(process_sequence)(sequence)
    for sequence in tqdm.tqdm(sequences))
total_count_array = sum(counts_array_lst)

# save the counts array
save_pickle('total_count_array.pkl', total_count_array)
