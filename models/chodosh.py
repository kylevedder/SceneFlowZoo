import numpy as np
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn

import numpy as np
from models.embedders import DynamicVoxelizer
from models.nsfp_baseline import NSFPProcessor
from models.nsfp import NSFPCached

from typing import Dict, Any, Optional
from collections import defaultdict
from pathlib import Path
import time


def fit_rigid(A, B):
    """
    Fit Rigid transformation A @ R.T + t = B
    """
    assert A.shape == B.shape
    num_rows, num_cols = A.shape

    # find mean column wise
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # ensure centroids are 1x3
    centroid_A = centroid_A.reshape(1, -1)
    centroid_B = centroid_B.reshape(1, -1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am.T @ Bm
    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -centroid_A @ R.T + centroid_B

    return R, t


def ransac_fit_rigid(pts1, pts2, inlier_thresh, ntrials):
    best_R = np.eye(3)
    best_t = np.zeros((3, ))
    best_inliers = (np.linalg.norm(pts1 - pts2, axis=-1) < inlier_thresh)
    best_inliers_sum = best_inliers.sum()
    for i in range(ntrials):
        choice = np.random.choice(len(pts1), 3)
        R, t = fit_rigid(pts1[choice], pts2[choice])
        inliers = (np.linalg.norm(pts1 @ R.T + t - pts2, axis=-1) <
                   inlier_thresh)
        if inliers.sum() > best_inliers_sum:
            best_R = R
            best_t = t
            best_inliers = inliers
            best_inliers_sum = best_inliers.sum()
            if best_inliers_sum / len(pts1) > 0.5:
                break
    best_R, best_t = fit_rigid(pts1[best_inliers], pts2[best_inliers])
    return best_R, best_t


def refine_flow(pc0,
                flow_pred,
                eps=0.4,
                min_points=10,
                motion_threshold=0.05,
                inlier_thresh=0.2,
                ntrials=250):
    labels = DBSCAN(eps=eps, min_samples=min_points).fit_predict(pc0)
    max_label = labels.max()
    refined_flow = np.zeros_like(flow_pred)
    for l in range(max_label + 1):
        label_mask = labels == l
        cluster_pts = pc0[label_mask]
        cluster_flows = flow_pred[label_mask]
        R, t = ransac_fit_rigid(cluster_pts, cluster_pts + cluster_flows,
                                inlier_thresh, ntrials)
        if np.linalg.norm(t) < motion_threshold:
            R = np.eye(3)
            t = np.zeros((3, ))
        refined_flow[label_mask] = (cluster_pts @ R.T + t) - cluster_pts

    refined_flow[labels == -1] = flow_pred[labels == -1]
    return refined_flow


class Chodosh(NSFPCached):

    def __init__(self, VOXEL_SIZE, POINT_CLOUD_RANGE, SEQUENCE_LENGTH,
                 nsfp_save_folder: Path, chodosh_save_folder: Path) -> None:
        super().__init__(VOXEL_SIZE, POINT_CLOUD_RANGE, SEQUENCE_LENGTH,
                         nsfp_save_folder)
        self.out_folder = Path(chodosh_save_folder)

    def forward(self, batched_sequence: Dict[str, torch.Tensor]):
        pc0_points_lst, pc0_valid_point_idxes, pc1_points_lst, pc1_valid_point_idxes = self._voxelize_batched_sequence(
            batched_sequence)

        log_ids_lst = self._transpose_list(batched_sequence['log_ids'])
        log_idxes_lst = self._transpose_list(batched_sequence['log_idxes'])

        flows = []
        delta_times = []
        for minibatch_idx, (pc0_points, pc0_valid_points_idx, log_ids,
                            log_idxes) in enumerate(
                                zip(pc0_points_lst, pc0_valid_point_idxes,
                                    log_ids_lst, log_idxes_lst)):
            assert len(
                log_ids) == 2, f"Expect 2 frames for NSFP, got {len(log_ids)}"
            assert len(
                log_idxes
            ) == 2, f"Expect 2 frames for NSFP, got {len(log_idxes)}"

            flow, flow_valid_idxes, delta_time = self._load_result(
                log_ids[0], log_idxes[0])

            assert flow_valid_idxes.shape == pc0_valid_points_idx.shape, f"{flow_valid_idxes.shape} != {pc0_valid_points_idx.shape}"
            idx_equality = flow_valid_idxes == pc0_valid_points_idx.detach(
            ).cpu().numpy()
            assert idx_equality.all(
            ), f"Points that disagree: {(~idx_equality).astype(int).sum()}"
            assert flow.shape[0] == 1, f"{flow.shape[0]} != 1"
            flow = flow.squeeze(0)
            assert flow.shape == pc0_points.shape, f"{flow.shape} != {pc0_points.shape}"

            breakpoint()

            flow[flow_valid_idxes] = refine_flow(
                pc0_points[pc0_valid_point_idxes].detach().cpu().numpy(),
                flow[flow_valid_idxes]).astype(np.float32)
            
            breakpoint()

            flows.append(torch.from_numpy(flow).to(pc0_points.device))
            delta_times.append(delta_time)

        return {
            "forward": {
                "flow": flows,
                "batch_delta_time": np.sum(delta_times),
                "pc0_points_lst": pc0_points_lst,
                "pc0_valid_point_idxes": pc0_valid_point_idxes,
                "pc1_points_lst": pc1_points_lst,
                "pc1_valid_point_idxes": pc1_valid_point_idxes
            }
        }