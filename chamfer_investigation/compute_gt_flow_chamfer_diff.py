import torch
import pandas as pd
import open3d as o3d
from dataloaders import ArgoverseRawSequenceLoader, ArgoverseSupervisedFlowSequenceLoader
from pointclouds import PointCloud, SE3
import numpy as np
import tqdm
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss import chamfer_distance

sequence_loader = ArgoverseSupervisedFlowSequenceLoader(
    '/efs/argoverse2/train/', '/efs/argoverse2/train_sceneflow/')


def pc_knn(warped_pc: PointCloud, target_pc: PointCloud, K: int) -> np.ndarray:

    warped_pc = torch.from_numpy(warped_pc.points).float().cuda().unsqueeze(0)
    target_pc = torch.from_numpy(target_pc.points).float().cuda().unsqueeze(0)

    # Compute min distance between warped point cloud and point cloud at t+1.
    knn = knn_points(p1=warped_pc, p2=target_pc, K=K, return_nn=True)

    return knn.knn.squeeze(0).cpu().numpy()


def stats_for_sequence(sequence, max_k=10, distance_threshold=2, sample_every=1):
    frame_list = sequence.load_frame_list(0)
    error_dicts = []
    idxes = list(range(len(frame_list) - 1))[::sample_every]
    for idx in tqdm.tqdm(idxes):

        frame_dict = frame_list[idx]
        pc = frame_dict['relative_pc']
        flowed_pc = frame_dict['relative_flowed_pc']
        classes = frame_dict['pc_classes']
        frame_dict_next = frame_list[idx + 1]
        pc_next = frame_dict_next['relative_pc']

        pc_full_knn_points = pc_knn(flowed_pc, pc_next, K=max_k)

        ground_truth_flows = flowed_pc.points - pc.points

        k_error_dict = {}

        for k in range(max_k):
            pc_k_knn_points = pc_full_knn_points[:, :k + 1].mean(axis=1)
            knn_flows = pc_k_knn_points - pc.points
            # Find points that are outside of the distance threshold.
            outside_distance = (np.linalg.norm(knn_flows, axis=1) >
                                distance_threshold)
            knn_flows[outside_distance] = 0
            # Compute the L2 endpoint error of the ground truth flow and the knn flow.
            endpoint_error = np.linalg.norm(ground_truth_flows - knn_flows,
                                            axis=1)

            # Split the endpoint error by class.
            class_error_dict = {}
            for cls_id in np.unique(classes):
                class_error_dict[cls_id] = endpoint_error[classes ==
                                                          cls_id].mean()

            # Split the endpoint error by gt speed.
            speed_error_dict = {}
            is_fast = np.linalg.norm(ground_truth_flows, axis=1) > 0.1
            speed_error_dict['fast'] = endpoint_error[is_fast].mean()
            speed_error_dict['slow'] = endpoint_error[~is_fast].mean()

            k_error_dict[k + 1] = {
                'class_error': class_error_dict,
                'speed_error': speed_error_dict,
                'endpoint_error': endpoint_error.mean(),
            }

        error_dicts.append(k_error_dict)
    return error_dicts


def compose_dicts(existing_dict, new_dict):
    for k, v in new_dict.items():
        if isinstance(v, dict):
            if k not in existing_dict:
                existing_dict[k] = {}
            existing_dict[k] = compose_dicts(existing_dict[k], v)
        else:
            assert isinstance(v, float), f"For Key {k}, Value {v} is not a float."
            if k not in existing_dict:
                existing_dict[k] = []
            existing_dict[k].append(v)
    return existing_dict

def merge_dicts(list_of_dicts):
    merged_dict = {}
    for d in list_of_dicts:
        merged_dict = compose_dicts(merged_dict, d)
    return merged_dict

sequence_dicts = []
for sequence_id in sequence_loader.get_sequence_ids()[:30]:
    print("Sequence ID: ", sequence_id)
    sequence = sequence_loader.load_sequence(sequence_id)
    sequence_dicts.extend(stats_for_sequence(sequence, sample_every=10))

merged_dicts = merge_dicts(sequence_dicts) 

# Save the merged dicts to a pickle file.
import pickle
with open('merged_dicts.pkl', 'wb') as f:
    pickle.dump(merged_dicts, f)