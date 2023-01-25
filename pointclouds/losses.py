import torch
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss import chamfer_distance
from pointclouds import PointCloud
from typing import Union


def warped_pc_loss(warped_pc: torch.Tensor,
                   target_pc: torch.Tensor,
                   dist_threshold=2.0):
    if warped_pc.ndim == 2:
        warped_pc = warped_pc.unsqueeze(0)
    if target_pc.ndim == 2:
        target_pc = target_pc.unsqueeze(0)

    loss = 0

    if dist_threshold is None:
        loss += chamfer_distance(warped_pc, target_pc,
                                 point_reduction="mean")[0].sum()
        loss += chamfer_distance(target_pc, warped_pc,
                                 point_reduction="mean")[0].sum()
        return loss

    # Compute min distance between warped point cloud and point cloud at t+1.

    warped_pc_shape_tensor = torch.LongTensor([warped_pc.shape[0]
                                               ]).to(warped_pc.device)
    target_pc_shape_tensor = torch.LongTensor([target_pc.shape[0]
                                               ]).to(target_pc.device)
    warped_to_target_knn = knn_points(p1=warped_pc,
                                      p2=target_pc,
                                      lengths1=warped_pc_shape_tensor,
                                      lengths2=target_pc_shape_tensor,
                                      K=1)
    warped_to_target_distances = warped_to_target_knn.dists[0]
    target_to_warped_knn = knn_points(p1=target_pc,
                                      p2=warped_pc,
                                      lengths1=target_pc_shape_tensor,
                                      lengths2=warped_pc_shape_tensor,
                                      K=1)
    target_to_warped_distances = target_to_warped_knn.dists[0]
    # Throw out distances that are too large (beyond the dist threshold).
    loss += warped_to_target_distances[
        warped_to_target_distances < dist_threshold].mean()

    loss += target_to_warped_distances[
        target_to_warped_distances < dist_threshold].mean()

    return loss


def pc0_to_pc1_distance(pc0: Union[PointCloud, torch.Tensor],
                        pc1: Union[PointCloud, torch.Tensor]):
    if isinstance(pc0, PointCloud):
        pc0 = pc0.points
    if isinstance(pc1, PointCloud):
        pc1 = pc1.points

    if pc0.ndim == 2:
        pc0 = pc0.unsqueeze(0)
    if pc1.ndim == 2:
        pc1 = pc1.unsqueeze(0)

    pc0_shape_tensor = torch.LongTensor([pc0.shape[0]]).to(pc0.device)
    pc1_shape_tensor = torch.LongTensor([pc1.shape[0]]).to(pc1.device)
    pc0_to_pc1_knn = knn_points(p1=pc0,
                                p2=pc1,
                                lengths1=pc0_shape_tensor,
                                lengths2=pc1_shape_tensor,
                                K=1)
    pc0_to_pc1_distances = pc0_to_pc1_knn.dists[0]
    return pc0_to_pc1_distances