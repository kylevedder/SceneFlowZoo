import torch
import FastGeodis
import torch.nn.functional as F

from dataclasses import dataclass
from .base_cost_function import BaseCostProblem
import numpy as np


class DistanceTransform:
    """
    Distance Transform implementation using FastGeodis taken from
    https://github.com/Lilac-Lee/FastNSF/blob/386ab3862be22a09542570abc7032e46fcea0802/optimization.py#L279-L330

    with some modifications to make it work more consistently with the rest of the codebase.
    """

    def __init__(
        self,
        pts: torch.Tensor,
        pmin: tuple[int, int, int],
        pmax: tuple[int, int, int],
        device: torch.device,
        grid_factor: float = 10.0,
    ):
        self.device = device
        self.grid_factor = grid_factor

        sample_x = ((pmax[0] - pmin[0]) * grid_factor).ceil().int() + 2
        sample_y = ((pmax[1] - pmin[1]) * grid_factor).ceil().int() + 2
        sample_z = ((pmax[2] - pmin[2]) * grid_factor).ceil().int() + 2

        self.Vx = (
            torch.linspace(0, sample_x, sample_x + 1, device=self.device)[:-1] / grid_factor
            + pmin[0]
        )
        self.Vy = (
            torch.linspace(0, sample_y, sample_y + 1, device=self.device)[:-1] / grid_factor
            + pmin[1]
        )
        self.Vz = (
            torch.linspace(0, sample_z, sample_z + 1, device=self.device)[:-1] / grid_factor
            + pmin[2]
        )

        # NOTE: build a binary image first, with 0-value occuppied points
        grid_x, grid_y, grid_z = torch.meshgrid(self.Vx, self.Vy, self.Vz, indexing="ij")
        self.grid = (
            torch.stack([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1), grid_z.unsqueeze(-1)], -1)
            .float()
            .squeeze()
        )
        H, W, D, _ = self.grid.size()
        pts_mask = torch.ones(H, W, D, device=device)
        self.pts_sample_idx_x = ((pts[:, 0:1] - self.Vx[0]) * self.grid_factor).round()
        self.pts_sample_idx_y = ((pts[:, 1:2] - self.Vy[0]) * self.grid_factor).round()
        self.pts_sample_idx_z = ((pts[:, 2:3] - self.Vz[0]) * self.grid_factor).round()
        pts_mask[
            self.pts_sample_idx_x.long(), self.pts_sample_idx_y.long(), self.pts_sample_idx_z.long()
        ] = 0.0

        iterations = 1
        image_pts = torch.zeros(H, W, D, device=device).unsqueeze(0).unsqueeze(0)
        pts_mask = pts_mask.unsqueeze(0).unsqueeze(0)
        self.D = FastGeodis.generalised_geodesic3d(
            image_pts,
            pts_mask,
            [1.0 / self.grid_factor, 1.0 / self.grid_factor, 1.0 / self.grid_factor],
            1e10,
            0.0,
            iterations,
        ).squeeze()
        breakpoint()

    @staticmethod
    def from_pointclouds(
        pc1: torch.Tensor, pc2: torch.Tensor, grid_factor: float = 10.0
    ) -> "DistanceTransform":
        assert (
            pc1.dim() == 3 and pc2.dim() == 3
        ), f"Input point clouds must be 3D: {pc1.dim()} != 3 or {pc2.dim()} != 3"
        # Must be same device
        assert (
            pc1.device == pc2.device
        ), f"Input point clouds must be on the same device: {pc1.device} != {pc2.device}"

        pc1_min = torch.min(pc1.squeeze(0), 0)[0]
        pc2_min = torch.min(pc2.squeeze(0), 0)[0]
        pc1_max = torch.max(pc1.squeeze(0), 0)[0]
        pc2_max = torch.max(pc2.squeeze(0), 0)[0]

        xmin_int, ymin_int, zmin_int = (
            torch.floor(torch.where(pc1_min < pc2_min, pc1_min, pc2_min) * grid_factor - 1)
            / grid_factor
        )
        xmax_int, ymax_int, zmax_int = (
            torch.ceil(torch.where(pc1_max > pc2_max, pc1_max, pc2_max) * grid_factor + 1)
            / grid_factor
        )

        return DistanceTransform(
            pc2.clone().squeeze(0).to(pc1.device),
            (xmin_int, ymin_int, zmin_int),
            (xmax_int, ymax_int, zmax_int),
            pc1.device,
            grid_factor,
        )

    @staticmethod
    def from_pointcloud(pc1: torch.Tensor, grid_factor: float = 10.0) -> "DistanceTransform":
        # If pc1 dim is 3, ensure that the first dim is a single dimension then squeeze it:
        if pc1.dim() == 3:
            assert (
                pc1.shape[0] == 1
            ), f"Input point clouds must have batch size of 1: {pc1.shape[0]}"
            pc1 = pc1.squeeze(0)

        assert pc1.dim() == 2, f"Input point clouds must be 3D: {pc1.dim()} != 3"

        pc1_min = torch.min(pc1, 0)[0]
        pc1_max = torch.max(pc1, 0)[0]

        xmin_int, ymin_int, zmin_int = torch.floor(pc1_min * grid_factor - 1) / grid_factor
        xmax_int, ymax_int, zmax_int = torch.ceil(pc1_max * grid_factor + 1) / grid_factor

        return DistanceTransform(
            pc1.clone().to(pc1.device),
            (xmin_int, ymin_int, zmin_int),
            (xmax_int, ymax_int, zmax_int),
            pc1.device,
            grid_factor,
        )

    def torch_bilinear_distance(self, Y: torch.Tensor):
        # Y can either by 1xNx3 or Nx3.
        if Y.dim() == 3:
            # Ensure that the first dim is a single dimension then squeeze it:
            assert Y.size(0) == 1, f"Input must have batch size of 1: {Y.size(0)}"
            Y = Y.squeeze(0)

        assert Y.dim() == 2, f"Input must be 2 dim: {Y.dim()}"
        assert Y.size(1) == 3, f"Input must have 3 columns: {Y.size(1)}"

        H, W, D = self.D.size()
        target = self.D[None, None, ...]

        sample_x = ((Y[:, 0:1] - self.Vx[0]) * self.grid_factor).clip(0, H - 1)
        sample_y = ((Y[:, 1:2] - self.Vy[0]) * self.grid_factor).clip(0, W - 1)
        sample_z = ((Y[:, 2:3] - self.Vz[0]) * self.grid_factor).clip(0, D - 1)

        sample = torch.cat([sample_x, sample_y, sample_z], -1)

        # NOTE: normalize samples to [-1, 1]
        sample = 2 * sample
        sample[..., 0] = sample[..., 0] / (H - 1)
        sample[..., 1] = sample[..., 1] / (W - 1)
        sample[..., 2] = sample[..., 2] / (D - 1)
        sample = sample - 1

        sample_ = torch.cat([sample[..., 2:3], sample[..., 1:2], sample[..., 0:1]], -1)

        # NOTE: reshape to match 5D volumetric input
        dist = F.grid_sample(
            target, sample_.view(1, -1, 1, 1, 3), mode="bilinear", align_corners=True
        ).view(-1)
        return dist

    def to_bev_image(self) -> np.ndarray:
        image_3d = self.D.cpu().numpy()
        return image_3d.mean(2)


@dataclass
class DistanceTransformLossProblem(BaseCostProblem):

    dt: DistanceTransform
    pc: torch.Tensor

    def __post_init__(self):
        assert self.pc.requires_grad, "pc must have requires_grad=True"

    def cost(self) -> torch.Tensor:
        res = self.dt.torch_bilinear_distance(self.pc).mean()
        assert res.requires_grad, "res must have requires_grad=True"
        return res
