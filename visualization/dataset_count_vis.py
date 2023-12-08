from pathlib import Path
from loader_utils import load_pickle
import numpy as np
from typing import Dict, List, Any

import matplotlib.pyplot as plt

BACKGROUND_CATEGORIES = ['BACKGROUND']

STUFF_CATEGORIES = [
    'BOLLARD',
    'CONSTRUCTION_BARREL',
    'CONSTRUCTION_CONE',
    'MOBILE_PEDESTRIAN_CROSSING_SIGN',
    'SIGN',
    'STOP_SIGN',
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
    "STUFF": STUFF_CATEGORIES,
    "PEDESTRIAN": PEDESTRIAN_CATEGORIES,
    "SMALL_MOVERS": SMALL_VEHICLE_CATEGORIES,
    "LARGE_MOVERS": VEHICLE_CATEGORIES
}

# Load the saved count data
raw_class_speed_info = load_pickle('dataset_count_info.pkl')


def merge_classes(class_a: Dict[float, int], class_b: Dict[float, int]):
    # Classes should have the same keys and the same number of keys
    assert class_a.keys() == class_b.keys()
    assert len(class_a.keys()) == len(class_b.keys())
    # Merge the two classes
    merged_class = {}
    for key in sorted(class_a.keys()):
        merged_class[key] = class_a[key] + class_b[key]
    return merged_class


def collapse_into_meta_classes(
        count_info: Dict[str, Dict[float,
                                   int]]) -> Dict[str, Dict[float, int]]:
    # Create a new dictionary to hold the meta classes
    meta_class_count_info = {}
    # For each meta class, merge the classes that belong to it
    for meta_class_name, meta_class_entries in METACATAGORIES.items():
        entries_list = [
            count_info[category_name] for category_name in meta_class_entries
        ]
        # Merge the classes
        merged_class = entries_list[0]
        for class_to_merge in entries_list[1:]:
            merged_class = merge_classes(merged_class, class_to_merge)
        # Add the merged class to the meta class dictionary
        meta_class_count_info[meta_class_name] = merged_class
    return meta_class_count_info


def collapse_speed_buckets(entry: Dict[float, int]) -> int:
    return sum(entry.values())


# Convert the array to a dictionary of

meta_class_speed_info = collapse_into_meta_classes(raw_class_speed_info)
meta_class_count_info = {
    meta_class_name: collapse_speed_buckets(meta_class_speed_info)
    for meta_class_name, meta_class_speed_info in
    meta_class_speed_info.items()
}

def plot_meta_class_counts(meta_class_count_info: Dict[str, int]):
    # Plot the counts
    fig, ax = plt.subplots()

    total_points = sum(meta_class_count_info.values())

    entries = sorted(meta_class_count_info.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*entries)
    
    bars = ax.bar(labels, values)
    ax.set_xlabel("Meta Class")
    ax.set_ylabel("Lidar points in semantic class")
    ax.set_title("Meta Class Counts")
    
    # Make the X axis labels at 30 Degrees
    plt.xticks(rotation=30, ha='right')
    # Make the Y axis log scale
    ax.set_yscale('log')

    # Annotate bars with percentage
    for bar in bars:
        height = bar.get_height()
        percentage = 100 * height / total_points
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{percentage:.2f}%', ha='center', va='bottom')

    # fig.savefig("meta_class_counts.pdf", bbox_inches='tight')
    # Save as a png as well
    fig.savefig("meta_class_counts.png", bbox_inches='tight')


plot_meta_class_counts(meta_class_count_info)

# category_normalized_count_array = count_array / count_array.sum(axis=1,
#                                                                 keepdims=True)

# ax = plt.gca()
# fig = plt.gcf()
# ax.matshow(category_normalized_count_array)
# for (i, j), z in np.ndenumerate(count_array):
#     ax.text(j,
#             i,
#             f'{z}',
#             ha='center',
#             va='center',
#             bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

# # Set X tick labels to be the speed bucket ranges, rotated 90 degrees
# ax.set_xticks(range(len(get_speed_bucket_ranges())),
#               [f"{l}-{u} m/s" for l, u in get_speed_bucket_ranges()],
#               rotation=90)
# # Set Y tick labels to be the category names
# ax.set_yticks(range(len(CATEGORY_ID_TO_IDX)), [
#     CATEGORY_ID_TO_NAME[CATEGORY_IDX_TO_ID[i]]
#     for i in range(len(CATEGORY_ID_TO_IDX))
# ])

# ax.set_xlabel('Speed Bucket Ranges')
# ax.set_ylabel('Categories')

# # Set figure to be 30x30
# fig.set_size_inches(30, 30)
# # Save the figure
# fig.savefig('total_count_array.png')
