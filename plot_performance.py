import numpy as np
from pathlib import Path
from loader_utils import load_pickle
import matplotlib
import matplotlib.pyplot as plt
import argparse
from typing import List, Tuple, Dict, Any, Optional
import shutil

CATEGORY_NAMES = {
    0: 'BACKGROUND',
    1: 'ANIMAL',
    2: 'ARTICULATED_BUS',
    3: 'BICYCLE',
    4: 'BICYCLIST',
    5: 'BOLLARD',
    6: 'BOX_TRUCK',
    7: 'BUS',
    8: 'CONSTRUCTION_BARREL',
    9: 'CONSTRUCTION_CONE',
    10: 'DOG',
    11: 'LARGE_VEHICLE',
    12: 'MESSAGE_BOARD_TRAILER',
    13: 'MOBILE_PEDESTRIAN_CROSSING_SIGN',
    14: 'MOTORCYCLE',
    15: 'MOTORCYCLIST',
    16: 'OFFICIAL_SIGNALER',
    17: 'PEDESTRIAN',
    18: 'RAILED_VEHICLE',
    19: 'REGULAR_VEHICLE',
    20: 'SCHOOL_BUS',
    21: 'SIGN',
    22: 'STOP_SIGN',
    23: 'STROLLER',
    24: 'TRAFFIC_LIGHT_TRAILER',
    25: 'TRUCK',
    26: 'TRUCK_CAB',
    27: 'VEHICULAR_TRAILER',
    28: 'WHEELCHAIR',
    29: 'WHEELED_DEVICE',
    30: 'WHEELED_RIDER'
}

CATEGORY_NAME_TO_IDX = {v: k for k, v in CATEGORY_NAMES.items()}

SPEED_BUCKET_SPLITS_METERS_PER_SECOND = [0, 0.1, 1.0, np.inf]

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

# Get path to methods from command line
parser = argparse.ArgumentParser()
parser.add_argument('results_folder', type=Path)
args = parser.parse_args()

assert args.results_folder.exists(
), f"Results folder {args.results_folder} does not exist"

save_folder = args.results_folder / "plots"
save_folder.mkdir(exist_ok=True, parents=True)


def set_font(size):
    matplotlib.rcParams.update({# Use mathtext, not LaTeX
                            'text.usetex': False,
                            # Use the Computer modern font
                            'font.family': 'serif',
                            'font.serif': ['cmr10'],
                            'font.size' : size,
                            'mathtext.fontset': 'cm',
                            # Use ASCII minus
                            'axes.unicode_minus': False,
                            })


def color(count, total_elements):
    start = 0.2
    stop = 0.8
    cm_subsection = np.linspace(start, stop, total_elements)
    return [matplotlib.cm.magma(x) for x in cm_subsection][count]


linewidth = 0.5
minor_tick_color = (0.9, 0.9, 0.9)

def grid():
    plt.grid(linewidth=linewidth / 2)
    plt.grid(which='minor',
             color=minor_tick_color,
             linestyle='--',
             alpha=0.7,
             clip_on=True,
             linewidth=linewidth / 4, zorder=0)


def savefig(name):
    for ext in ['pdf', 'png']:
        plt.savefig(save_folder / f"{name}.{ext}", bbox_inches='tight')
    plt.clf()


class ResultInfo():

    def __init__(self, name, path: Path):
        self.name = name
        self.path = path
        self.results = load_pickle(path)

    def __repr__(self):
        return self.name + " @ " + str(self.path)

    def pretty_name(self):
        name_dict = {
            "fastflow3d_nsfp_distilatation": "DRLFS (Ours)",
            "fastflow3d_supervised": "FastFlow3D Supervised",
            "nsfp_unsupervised_flow_data_cached": "NSFP",
        }
        if self.name in name_dict:
            return name_dict[self.name]
        print("WARNING: No pretty name for", self.name)
        return self.name

    def speed_bucket_categories(self):
        return list(
            zip(SPEED_BUCKET_SPLITS_METERS_PER_SECOND,
                SPEED_BUCKET_SPLITS_METERS_PER_SECOND[1:]))

    def get_metacatagory_epe_by_speed(self, metacatagory):
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
            return [np.sum(error_sum) / np.sum(error_count)]
        return error_sum / error_count


def load_results(validation_folder: Path):
    print("Loading results from", validation_folder)
    config_folder = validation_folder / "configs"
    print()
    assert config_folder.exists(
    ), f"Config folder {config_folder} does not exist"
    result_lst = []
    for architecture_folder in sorted(config_folder.glob("*/")):
        for result_file in architecture_folder.glob("*.pkl"):
            result_lst.append(
                ResultInfo(
                    architecture_folder.name + "_" +
                    result_file.stem.split(".")[0], result_file))

    return result_lst


print("Loading results...")
results = load_results(args.results_folder)
print("Done loading results.")
print(results)


def process_metacategory_counts(result):
    full_error_count = result['per_class_bucketed_error_count']
    metacatagory_results = {}
    # How do we do on vehicles by speed?
    for metacatagory in METACATAGORIES:
        category_names = METACATAGORIES[metacatagory]
        category_idxes = [CATEGORY_NAME_TO_IDX[cat] for cat in category_names]

        metacatagory_counts = full_error_count[category_idxes]
        # Sum up other axes
        metacatagory_counts = np.sum(metacatagory_counts, axis=(1, 2))
        metacatagory_results[metacatagory] = {}
        for category_result, category_name in zip(metacatagory_counts,
                                                  category_names):
            metacatagory_results[metacatagory][category_name] = category_result

    return metacatagory_results


def process_category_speed_counts(result):
    category_speed_counts_raw = result['per_class_bucketed_error_count'].sum(
        axis=2)
    category_speed_counts_normalized = category_speed_counts_raw / category_speed_counts_raw.sum(
        axis=1, keepdims=True)
    return category_speed_counts_normalized, category_speed_counts_raw


def bar_offset(pos_idx, num_pos, position_offset=0.2):
    """
    Compute the X offset for a bar plot given the number of bars and the
    index of the bar.
    """
    return -(num_pos + 1) / 2 * position_offset + position_offset * (pos_idx +
                                                                     1)


BAR_WIDTH = 0.2

num_metacatagories = len(METACATAGORIES)


def merge_dict_list(dict_list):
    result_dict = {}
    for d in dict_list:
        for k, v in d.items():
            if k not in result_dict:
                result_dict[k] = []
            result_dict[k].append(v)
    return result_dict


def plot_metacatagory_speed_vs_error(results: List[ResultInfo], metacatagory,
                                     vmax):
    for result_idx, result in enumerate(results):
        metacatagory_epe_by_speed = result.get_metacatagory_epe_by_speed(
            metacatagory)
        xs = np.arange(len(metacatagory_epe_by_speed)) + bar_offset(
            result_idx, len(results), BAR_WIDTH)
        plt.bar(xs,
                metacatagory_epe_by_speed,
                label=result.pretty_name(),
                width=BAR_WIDTH,
                color=color(result_idx, len(results)),
                zorder=3)
    if metacatagory == "BACKGROUND":
        plt.xticks([-1, 0, 1], ["", "All", ""])
        plt.xlabel(" ")
        plt.legend()
    else:
        speed_buckets = result.speed_bucket_categories()
        plt.xticks(np.arange(len(speed_buckets)),
                   [f"{l:0.1f}-" + (f"{u:0.1f}" if u != np.inf else "$\infty$") for l, u in speed_buckets],
                   rotation=0)
        plt.xlabel("Speed Bucket (m/s)")
    plt.ylabel("EPE (m)")
    plt.ylim(0, vmax)
    grid()


set_font(8)

for metacatagory in METACATAGORIES:
    plt.gcf().set_size_inches(5.5 / 2, 5.5 / 1.6 / 2)
    plot_metacatagory_speed_vs_error(results, metacatagory, vmax=0.25)
    savefig(f"speed_vs_error_{metacatagory}")

# def plot_meta_catagory_category_counts():
#     fig, subplot_axes = plt.subplots(1, num_metacatagories)

#     for result_idx, result in enumerate(results):
#         metacategory_counts = process_metacategory_counts(result.results)

#         for metacatagory, ax in zip(metacategory_counts, subplot_axes):
#             category_counts = metacategory_counts[metacatagory]
#             categories, counts = zip(*category_counts.items())
#             ax.set_title(metacatagory)
#             ax.bar(
#                 np.arange(len(categories)),
#                 counts,
#                 label=result.name.replace("fastflow3d_", " "),
#             )
#             ax.set_xticks(range(len(categories)), categories, rotation=90)

#     return fig

# def plot_category_speed_counts():
#     fig, axes = plt.subplots(1, len(results))
#     for result_idx, (result, ax) in enumerate(zip(results, axes)):
#         speed_counts_norm, speed_counts_raw = process_category_speed_counts(
#             result.results)
#         ax.matshow(speed_counts_norm)
#         for (i, j), z in np.ndenumerate(speed_counts_raw):
#             ax.text(j,
#                     i,
#                     '{}'.format(z),
#                     ha='center',
#                     va='center',
#                     bbox=dict(boxstyle='round',
#                               facecolor='white',
#                               edgecolor='0.3'))
#         ax.set_yticks(range(len(CATEGORY_NAMES)), CATEGORY_NAMES.values())

#     return fig

# fig = plot_meta_catagory_category_counts()
# fig.set_size_inches(20, 10)
# plt.tight_layout()
# fig.savefig(args.results_folder / "category_counts.png")
# plt.clf()

# fig = plot_category_speed_counts()
# fig.set_size_inches(10, 30)
# plt.tight_layout()
# fig.savefig(args.results_folder / "category_speed_counts.png")
# plt.clf()
