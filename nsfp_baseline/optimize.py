import copy
import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds

from pointclouds import warped_pc_loss

from .utils import EarlyStopping
from .model import Neural_Prior

from typing import Union

import torch
import torch.nn as nn


def _validate_chamfer_reduction_inputs(batch_reduction: Union[str, None],
                                       point_reduction: str):
    """Check the requested reductions are valid.
    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError(
            'batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')


def _handle_pointcloud_input(
    points: Union[torch.Tensor, Pointclouds],
    lengths: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError(
                f"Expected points to be of shape (N, P, D), got {points.shape}"
            )
        X = points
        if lengths is not None and (lengths.ndim != 1
                                    or lengths.shape[0] != X.shape[0]):
            raise ValueError("Expected lengths to be of shape (N,)")
        if lengths is None:
            lengths = torch.full((X.shape[0], ),
                                 X.shape[1],
                                 dtype=torch.int64,
                                 device=points.device)
    else:
        raise ValueError("The input pointclouds should be either " +
                         "Pointclouds objects or torch.Tensor of shape " +
                         "(minibatch, num_points, 3).")
    return X, lengths


def my_chamfer_fn(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    assert torch.isfinite(x).all(), "x contains NaNs"
    assert torch.isfinite(y).all(), "y contains NaNs"

    x, x_lengths = _handle_pointcloud_input(x, None)
    y, y_lengths = _handle_pointcloud_input(y, None)

    if x_lengths.item() == 0:
        breakpoint()

    assert x_lengths.item() > 0, f"x_lengths is {x_lengths.item()}"
    assert y_lengths.item() > 0, f"y_lengths is {y_lengths.item()}"

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
              )  # shape [N, P1]
    y_mask = (torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
              )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum(
                (1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    assert torch.isfinite(cham_x).all(), "cham_x contains NaNs"
    assert torch.isfinite(cham_y).all(), "cham_y contains NaNs"

    # NOTE: truncated Chamfer distance.
    dist_thd = 2
    x_mask[cham_x >= dist_thd] = True
    y_mask[cham_y >= dist_thd] = True
    cham_x[x_mask] = 0.0
    cham_y[y_mask] = 0.0

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6))
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6))

        if is_x_heterogeneous:
            # pyre-fixme[16]: `int` has no attribute `__setitem__`.
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)

    assert torch.isfinite(cham_x).all(), "cham_x contains NaNs"
    assert torch.isfinite(cham_y).all(), "cham_y contains NaNs"

    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths

    assert torch.isfinite(cham_x).all(), "cham_x contains NaNs"
    assert torch.isfinite(cham_y).all(), "cham_y contains NaNs"

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if batch_reduction == "mean":
            cham_x /= N
            cham_y /= N

    assert torch.isfinite(cham_x).all(), "cham_x contains NaNs"
    assert torch.isfinite(cham_y).all(), "cham_y contains NaNs"
    cham_dist = cham_x + cham_y
    cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    return cham_dist, cham_normals


def optimize(pc1, pc2, iterations=5000, lr=0.008):

    net = Neural_Prior(filter_size=128, act_fn='relu', layer_size=8).cuda()
    early_stopping = EarlyStopping(patience=100, min_delta=0.0001)

    for param in net.parameters():
        param.requires_grad = True

    net_inv = copy.deepcopy(net)
    params = [{
        'params': net.parameters(),
        'lr': lr,
        'weight_decay': 0
    }, {
        'params': net_inv.parameters(),
        'lr': lr,
        'weight_decay': 0
    }]
    best_forward = {'loss': torch.inf}
    optimizer = torch.optim.Adam(params, lr=0.008, weight_decay=0)
    for epoch in range(iterations):
        optimizer.zero_grad()

        forward_flow = net(pc1)
        pc1_warped_to_pc2 = pc1 + forward_flow
        forward_loss, _ = my_chamfer_fn(pc1_warped_to_pc2, pc2, None, None)
        assert torch.isfinite(
            forward_loss).all(), f'forward_loss: {forward_loss.item()}'

        print(f'epoch: {epoch}, forward_loss: {forward_loss.item()}')

        reverse_flow = net_inv(pc1_warped_to_pc2)
        est_pc2_warped_to_pc1 = pc1_warped_to_pc2 - reverse_flow
        reversed_loss, _ = my_chamfer_fn(est_pc2_warped_to_pc1, pc1, None,
                                         None)
        assert torch.isfinite(
            reversed_loss).all(), f'reversed_loss: {reversed_loss.item()}'

        loss_chamfer = forward_loss + reversed_loss

        loss = loss_chamfer

        # ANCHOR: get best metrics
        if forward_loss <= best_forward['loss']:
            best_forward['loss'] = forward_loss.item()
            best_forward['warped_pc'] = pc1_warped_to_pc2
            best_forward['target_pc'] = pc2

        if early_stopping.step(loss):
            break

        loss.backward()
        optimizer.step()
    return best_forward


class NSFPProcessor(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pc1, pc2):
        res = optimize(pc1, pc2)
        return res['warped_pc'], res['target_pc']


class NSFPLoss():

    def __init__(self):
        pass

    def __call__(self, warped_pc, target_pc):
        return warped_pc_loss(warped_pc, target_pc)
