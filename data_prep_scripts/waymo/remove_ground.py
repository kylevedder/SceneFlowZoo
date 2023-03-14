import sys

sys.path.append('.')

import torch

from loader_utils import save_npy

from dataloaders import WaymoRawSequence, WaymoRawSequenceLoader
from pointclouds import PointCloud, SE3
import numpy as np
import tqdm
import argparse
from pathlib import Path
import collections
import open3d as o3d
import copy

# Get path to waymo data from command line
parser = argparse.ArgumentParser()
parser.add_argument('path', type=Path)
args = parser.parse_args()

assert args.path.exists(), f"Path {args.path} does not exist"


class GroundRemoverNet(torch.nn.Module):
    """
    3 layer coordinate network of 16 hidden units each
    """

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

        self.huber_loss = torch.nn.HuberLoss(delta=0.1)

    def forward(self, pc_points: torch.Tensor):
        """
        Args:
            pc: PointCloud of shape (N, 3)
        Returns:
            logits: Tensor of shape (N, 1)
        """
        xy_points = pc_points[:, :2]
        res = self.net(xy_points)
        return res.squeeze(1)

    def loss(self, h, z, c=0.3):
        case_1 = (h - z)**2
        case_2 = self.huber_loss(h, z)
        case_3 = torch.ones_like(h) * c

        case_1_mask = (z < h)
        case_2_mask = ((h <= z) & (z < h + c))

        return torch.mean(
            torch.where(case_1_mask, case_1,
                        torch.where(case_2_mask, case_2, case_3)))


def process_sequence(sequence: WaymoRawSequence):

    for idx, entry_dict in enumerate(tqdm.tqdm(
            sequence.load_frame_list(None))):
        pc = entry_dict['relative_pc']
        pose = entry_dict['relative_pose']

        net = GroundRemoverNet().cuda()
        # setup gradient descent optimizer
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=0.01,
                                     weight_decay=0.0)
        schedule = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=750,
                                                   gamma=0.5)
        max_iterations = 5000
        best_loss = None
        best_params = None
        best_idx = None

        c = 0.3

        pc = torch.from_numpy(pc.points).float().cuda()
        for i in tqdm.tqdm(range(max_iterations)):
            optimizer.zero_grad()
            logits = net(pc)
            loss = net.loss(logits, pc[:, 2], c=c)
            loss.backward()
            optimizer.step()
            schedule.step()

            loss_item = loss.item()

            if best_loss is None or loss_item < best_loss:
                best_loss = loss_item
                best_params = copy.deepcopy(net.state_dict())
                best_idx = i

        net.load_state_dict(best_params)
        print("Best idx:", best_idx)

        # Points where z <= net(x, y) + c are ground
        z_points = pc[:, 2]
        ground_mask = ((net(pc) + c) >= z_points)

        # Use open3d to visualize the pointcloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.cpu().numpy())
        # Use the ground mask to color the points
        pcd.colors = o3d.utility.Vector3dVector(
            np.array([[1, 0, 0],
                      [0, 1, 0]])[ground_mask.cpu().numpy().astype(np.int)])

        o3d.visualization.draw_geometries([pcd])

        save_path = args.path / 'ground_removed' / sequence.sequence_name / f'{idx:06d}.npy'
        save_npy(save_path, ground_mask.cpu().numpy())


sequence_loader = WaymoRawSequenceLoader(args.path)
for sequence_id in sequence_loader.get_sequence_ids():
    process_sequence(sequence_loader.load_sequence(sequence_id))
