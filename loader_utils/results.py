import numpy as np
from pathlib import Path
from .loaders import load_pickle
from typing import List, Dict, Tuple, Union, Optional, Any
import copy
import enum

CATEGORY_ID_TO_NAME = {
    -1: 'BACKGROUND',
    0: 'ANIMAL',
    1: 'ARTICULATED_BUS',
    2: 'BICYCLE',
    3: 'BICYCLIST',
    4: 'BOLLARD',
    5: 'BOX_TRUCK',
    6: 'BUS',
    7: 'CONSTRUCTION_BARREL',
    8: 'CONSTRUCTION_CONE',
    9: 'DOG',
    10: 'LARGE_VEHICLE',
    11: 'MESSAGE_BOARD_TRAILER',
    12: 'MOBILE_PEDESTRIAN_CROSSING_SIGN',
    13: 'MOTORCYCLE',
    14: 'MOTORCYCLIST',
    15: 'OFFICIAL_SIGNALER',
    16: 'PEDESTRIAN',
    17: 'RAILED_VEHICLE',
    18: 'REGULAR_VEHICLE',
    19: 'SCHOOL_BUS',
    20: 'SIGN',
    21: 'STOP_SIGN',
    22: 'STROLLER',
    23: 'TRAFFIC_LIGHT_TRAILER',
    24: 'TRUCK',
    25: 'TRUCK_CAB',
    26: 'VEHICULAR_TRAILER',
    27: 'WHEELCHAIR',
    28: 'WHEELED_DEVICE',
    29: 'WHEELED_RIDER'
}

CATEGORY_NAME_TO_IDX = {
    v: idx
    for idx, (_, v) in enumerate(sorted(CATEGORY_ID_TO_NAME.items()))
}

SPEED_BUCKET_SPLITS_METERS_PER_SECOND = [0, 0.5, 2.0, np.inf]
ENDPOINT_ERROR_SPLITS_METERS = [0, 0.05, 0.1, np.inf]

BACKGROUND_CATEGORIES = [
    'BOLLARD', 'CONSTRUCTION_BARREL', 'CONSTRUCTION_CONE',
    'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'SIGN', 'STOP_SIGN'
]
PEDESTRIAN_CATEGORIES = [
    'PEDESTRIAN', 'STROLLER', 'WHEELCHAIR', 'OFFICIAL_SIGNALER'
]
SMALL_VEHICLE_CATEGORIES = [
    'BICYCLE', 'BICYCLIST', 'MOTORCYCLE', 'MOTORCYCLIST', 'WHEELED_DEVICE',
    'WHEELED_RIDER'
]
VEHICLE_CATEGORIES = [
    'ARTICULATED_BUS', 'BOX_TRUCK', 'BUS', 'LARGE_VEHICLE', 'RAILED_VEHICLE',
    'REGULAR_VEHICLE', 'SCHOOL_BUS', 'TRUCK', 'TRUCK_CAB', 'VEHICULAR_TRAILER',
    'TRAFFIC_LIGHT_TRAILER', 'MESSAGE_BOARD_TRAILER'
]
ANIMAL_CATEGORIES = ['ANIMAL', 'DOG']

METACATAGORIES = {
    "BACKGROUND": BACKGROUND_CATEGORIES,
    "PEDESTRIAN": PEDESTRIAN_CATEGORIES,
    "SMALL_MOVERS": SMALL_VEHICLE_CATEGORIES,
    "LARGE_MOVERS": VEHICLE_CATEGORIES
}

METACATAGORY_TO_SHORTNAME = {
    "BACKGROUND": "BG",
    "PEDESTRIAN": "PED",
    "SMALL_MOVERS": "SMALL",
    "LARGE_MOVERS": "LARGE"
}


# Enum for close, far, and full
class DistanceThreshold(enum.Enum):
    CLOSE = 0
    FAR = 1
    ALL = 2


class ResultInfo():

    def __init__(self,
                 name,
                 path: Union[Path, dict],
                 full_distance: DistanceThreshold = DistanceThreshold.ALL):
        self.name = name
        self.path = path
        if isinstance(full_distance, str):
            full_distance = DistanceThreshold[full_distance.upper()]
        if isinstance(path, dict):
            assert 'per_class_bucketed_error_sum' in path, f"Path {path} does not contain per_class_bucketed_error_sum"
            assert 'per_class_bucketed_error_count' in path, f"Path {path} does not contain per_class_bucketed_error_count"
            self.results = copy.deepcopy(path)
        else:
            self.results = load_pickle(path)

        # Shapes should be (distance threshold, num_classes, num_speed_buckets, num_endpoint_error_buckets)
        assert self.results[
            'per_class_bucketed_error_sum'].ndim == 4, f"per_class_bucketed_error_sum should have 4 dimensions but has {self.results['per_class_bucketed_error_sum'].ndim}; shape is {self.results['per_class_bucketed_error_sum'].shape}"
        assert self.results[
            'per_class_bucketed_error_count'].ndim == 4, f"per_class_bucketed_error_count should have 4 dimensions but has {self.results['per_class_bucketed_error_count'].ndim}; shape is {self.results['per_class_bucketed_error_count'].shape}"

        self.full_distance = full_distance
        if full_distance == DistanceThreshold.ALL:
            # We need to sum the first axis of the sum and count arrays
            self.results['per_class_bucketed_error_sum'] = np.sum(
                self.results['per_class_bucketed_error_sum'], axis=0)
            self.results['per_class_bucketed_error_count'] = np.sum(
                self.results['per_class_bucketed_error_count'], axis=0)
        elif full_distance == DistanceThreshold.CLOSE:
            # We need to select the first element of the sum and count arrays
            self.results['per_class_bucketed_error_sum'] = self.results[
                'per_class_bucketed_error_sum'][0]
            self.results['per_class_bucketed_error_count'] = self.results[
                'per_class_bucketed_error_count'][0]
        elif full_distance == DistanceThreshold.FAR:
            # We need to select the second element of the sum and count arrays
            self.results['per_class_bucketed_error_sum'] = self.results[
                'per_class_bucketed_error_sum'][1]
            self.results['per_class_bucketed_error_count'] = self.results[
                'per_class_bucketed_error_count'][1]
        else:
            raise ValueError(f"Unknown distance threshold {full_distance}")

        assert self.results['per_class_bucketed_error_sum'].shape == (
            len(CATEGORY_ID_TO_NAME),
            len(SPEED_BUCKET_SPLITS_METERS_PER_SECOND) - 1,
            len(ENDPOINT_ERROR_SPLITS_METERS) - 1
        ), f"Shape of per_class_bucketed_error_sum is {self.results['per_class_bucketed_error_sum'].shape} but should be {(len(CATEGORY_ID_TO_NAME), len(SPEED_BUCKET_SPLITS_METERS_PER_SECOND) - 1, len(ENDPOINT_ERROR_SPLITS_METERS) - 1)}"
        assert self.results['per_class_bucketed_error_count'].shape == (
            len(CATEGORY_ID_TO_NAME),
            len(SPEED_BUCKET_SPLITS_METERS_PER_SECOND) - 1,
            len(ENDPOINT_ERROR_SPLITS_METERS) - 1
        ), f"Shape of per_class_bucketed_error_count is {self.results['per_class_bucketed_error_count'].shape} but should be {(len(CATEGORY_ID_TO_NAME), len(SPEED_BUCKET_SPLITS_METERS_PER_SECOND) - 1, len(ENDPOINT_ERROR_SPLITS_METERS) - 1)}"

    def __repr__(self):
        return self.name + " @ " + str(self.path)

    def pretty_name(self):
        name_dict = {
            "fastflow3d_nsfp_distilatation": "DRLFS (Ours)",
            "fastflow3d_nsfp_distilatation_2x": "DRLFS 2x (Ours)",
            "fastflow3d_nsfp_distilatation_0.5x": "DRLFS 0.5x (Ours)",
            "fastflow3d_nsfp_distilatation_0.1x": "DRLFS 0.1x (Ours)",
            "fastflow3d_nsfp_distilatation_speed_scaled": "DRLFS (Ours, speed scaled)",
            "fastflow3d_supervised": "FastFlow3D",
            "fastflow3d_chodosh_distilatation": "DRLFS (Chodosh et al. pseudolabels)",
            "fastflow3d_supervised_no_scale": "FastFlow3D (no scale)",
            "nsfp_unsupervised_sensor_val_cached": "NSFP",
        }
        if self.name in name_dict:
            return name_dict[self.name]
        print("WARNING: No pretty name for", self.name)
        return self.name

    def speed_bucket_categories(self):
        return list(
            zip(SPEED_BUCKET_SPLITS_METERS_PER_SECOND,
                SPEED_BUCKET_SPLITS_METERS_PER_SECOND[1:]))

    def endpoint_error_categories(self):
        return list(
            zip(ENDPOINT_ERROR_SPLITS_METERS,
                ENDPOINT_ERROR_SPLITS_METERS[1:]))

    def get_metacatagory_error_count_by_speed(self, metacatagory):
        full_error_sum = self.results['per_class_bucketed_error_sum']
        full_error_count = self.results['per_class_bucketed_error_count']
        category_idxes = [
            CATEGORY_NAME_TO_IDX[cat] for cat in METACATAGORIES[metacatagory]
        ]
        error_sum = full_error_sum[category_idxes]
        error_count = full_error_count[category_idxes]
        # Sum up all the catagories into a single metacatagory
        error_sum = np.sum(error_sum, axis=0)
        error_count = np.sum(error_count, axis=0)
        # Sum up all the EPEs
        error_sum = np.sum(error_sum, axis=1)
        error_count = np.sum(error_count, axis=1)
        # Only remaining axis is speed
        if metacatagory == "BACKGROUND":
            return np.array([np.sum(error_sum)
                             ]), np.array([np.sum(error_count)])
        return error_sum, error_count

    def get_metacatagory_error_count_by_speed_and_epe(self, metacatagory):
        full_error_sum = self.results['per_class_bucketed_error_sum']
        full_error_count = self.results['per_class_bucketed_error_count']
        category_idxes = [
            CATEGORY_NAME_TO_IDX[cat] for cat in METACATAGORIES[metacatagory]
        ]
        error_sum = full_error_sum[category_idxes]
        error_count = full_error_count[category_idxes]
        # Sum up all the catagories into a single metacatagory
        error_sum = np.sum(error_sum, axis=0)
        error_count = np.sum(error_count, axis=0)
        # Only remaining axis is speed
        if metacatagory == "BACKGROUND":
            return np.array([np.sum(error_sum, axis=0)
                             ]), np.array([np.sum(error_count, axis=0)])
        return error_sum, error_count

    def get_metacatagory_epe_by_speed(self, metacatagory):
        error_sum, error_count = self.get_metacatagory_error_count_by_speed(
            metacatagory)
        return error_sum / error_count

    def get_mover_point_all_epe(self):
        mover_metacatagories = set(METACATAGORIES.keys()) - {"BACKGROUND"}
        total_error = 0
        total_count = 0

        # Sum up the moving components of the movers
        for metacatagory in mover_metacatagories:
            error_sum, error_count = self.get_metacatagory_error_count_by_speed(
                metacatagory)
            # Extract the non-background error
            total_error += np.sum(error_sum)
            total_count += np.sum(error_count)

        return total_error / total_count

    def get_mover_point_dynamic_epe(self):
        mover_metacatagories = set(METACATAGORIES.keys()) - {"BACKGROUND"}
        total_error = 0
        total_count = 0

        # Sum up the moving components of the movers
        for metacatagory in mover_metacatagories:
            error_sum, error_count = self.get_metacatagory_error_count_by_speed(
                metacatagory)
            # Extract the non-background error
            total_error += np.sum(error_sum[1:])
            total_count += np.sum(error_count[1:])

        return total_error / total_count

    def get_mover_point_static_epe(self):
        mover_metacatagories = set(METACATAGORIES.keys()) - {"BACKGROUND"}
        total_error = 0
        total_count = 0

        # Sum up the moving components of the movers
        for metacatagory in mover_metacatagories:
            error_sum, error_count = self.get_metacatagory_error_count_by_speed(
                metacatagory)
            # Extract the non-background error
            total_error += error_sum[0]
            total_count += error_count[0]

        return total_error / total_count

    def get_nonmover_point_epe(self):
        mover_metacatagories = set(METACATAGORIES.keys()) - {"BACKGROUND"}

        total_error = 0
        total_count = 0

        # Sum up the stationary components of the movers
        for metacatagory in mover_metacatagories:
            error_sum, error_count = self.get_metacatagory_error_count_by_speed(
                metacatagory)
            total_error += error_sum[0]
            total_count += error_count[0]

        error_sum, error_count = self.get_metacatagory_error_count_by_speed(
            "BACKGROUND")
        total_error += np.sum(error_sum)
        total_count += np.sum(error_count)

        return total_error / total_count

    def get_metacatagory_count_by_epe(self, metacatagory):
        full_error_count = self.results['per_class_bucketed_error_count']
        category_idxes = [
            CATEGORY_NAME_TO_IDX[cat] for cat in METACATAGORIES[metacatagory]
        ]
        error_count = full_error_count[category_idxes]
        # Sum up all the catagories into a single metacatagory
        error_count = np.sum(error_count, axis=0)
        # Sum up all the speeds
        error_count = np.sum(error_count, axis=0)
        # Only remaining axis is epe
        return error_count

    def get_foreground_counts(self):
        total_epe_count = np.zeros((len(ENDPOINT_ERROR_SPLITS_METERS) - 1))
        mover_metacatagories = set(METACATAGORIES.keys()) - {"BACKGROUND"}
        for metacatagory in mover_metacatagories:
            _, error_count = self.get_metacatagory_error_count_by_speed_and_epe(
                metacatagory)
            # Extract the non-background error for all EPE buckets
            total_epe_count += np.sum(error_count[1:], axis=0)
        return total_epe_count

    def get_metacatagory_count_by_epe_strict(self, metacatagory):
        count_by_epe = self.get_metacatagory_count_by_epe(metacatagory)
        return count_by_epe[0]

    def get_metacatagory_count_by_epe_relaxed(self, metacatagory):
        count_by_epe = self.get_metacatagory_count_by_epe(metacatagory)
        return np.sum(count_by_epe[:2])

    def get_metacatagory_count_by_epe_all(self, metacatagory):
        count_by_epe = self.get_metacatagory_count_by_epe(metacatagory)
        return np.sum(count_by_epe)

    def get_latency(self):
        return self.results['average_forward_time']
