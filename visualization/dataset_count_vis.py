from pathlib import Path
from loader_utils import load_pickle
import numpy as np

import matplotlib.pyplot as plt

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


# Load the saved count data
count_array = load_pickle('total_count_array.pkl')

category_normalized_count_array = count_array / count_array.sum(axis=1,
                                                                keepdims=True)

ax = plt.gca()
fig = plt.gcf()
ax.matshow(category_normalized_count_array)
for (i, j), z in np.ndenumerate(count_array):
    ax.text(j,
            i,
            f'{z}',
            ha='center',
            va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

# Set X tick labels to be the speed bucket ranges, rotated 90 degrees
ax.set_xticks(range(len(get_speed_bucket_ranges())),
              [f"{l}-{u} m/s" for l, u in get_speed_bucket_ranges()],
              rotation=90)
# Set Y tick labels to be the category names
ax.set_yticks(range(len(CATEGORY_ID_TO_IDX)), [
    CATEGORY_ID_TO_NAME[CATEGORY_IDX_TO_ID[i]]
    for i in range(len(CATEGORY_ID_TO_IDX))
])

ax.set_xlabel('Speed Bucket Ranges')
ax.set_ylabel('Categories')

# Set figure to be 30x30
fig.set_size_inches(30, 30)
# Save the figure
fig.savefig('total_count_array.png')
