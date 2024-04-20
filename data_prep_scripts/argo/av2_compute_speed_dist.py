from bucketed_scene_flow_eval.datasets.argoverse2 import ArgoverseSceneFlowSequenceLoader

from bucketed_scene_flow_eval.datastructures import TimeSyncedSceneFlowFrame, TimeSyncedAVLidarData

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import joblib

sequence_loader = ArgoverseSceneFlowSequenceLoader(
    "/efs/argoverse2/val/",
    "/efs/argoverse2/val_sceneflow_feather/",
    with_rgb=False,
)


all_speeds = []


def process_sequence(sequence_id: str):
    all_speeds = []
    sequence = sequence_loader.load_sequence(sequence_id)
    for idx in range(len(sequence) - 1):
        flow_frame, _ = sequence.load(idx, relative_to_idx=0, with_flow=True)
        valid_flow = flow_frame.flow.valid_flow
        speeds = np.linalg.norm(valid_flow, axis=1)
        all_speeds.extend(speeds)

    return all_speeds


# results = [
#     process_sequence(sequence_id) for sequence_id in tqdm.tqdm(sequence_loader.get_sequence_ids())
# ]

# # Parallelize with joblib
# results = joblib.Parallel(n_jobs=36)(
#     joblib.delayed(process_sequence)(sequence_id)
#     for sequence_id in tqdm.tqdm(sequence_loader.get_sequence_ids())
# )

all_speeds = np.array([])
for sequence_id in tqdm.tqdm(sequence_loader.get_sequence_ids()):
    all_speeds = np.append(all_speeds, process_sequence(sequence_id))


# Compute 99th, 99.9th, 99.99th percentiles
percentiles = np.percentile(all_speeds, [99, 99.9, 99.99])
print("99th percentile: ", percentiles[0])
print("99.9th percentile: ", percentiles[1])
print("99.99th percentile: ", percentiles[2])


plt.hist(all_speeds, bins=100)
plt.xlabel("Speed (m/s)")
plt.ylabel("Frequency")
plt.title("Speed distribution of Argoverse2 validation set")
# Y log scale
plt.yscale("log")
plt.show()
