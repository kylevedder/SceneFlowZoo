import torch
from pointclouds import PointCloud, SE3


class LoaderToTorch():

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __get_item__(self, index: int):
        pc, pose = self.loader[index]
        pc_points = torch.from_numpy(pc.to_array()).float().to(self.device)
        pose_matrix = torch.from_numpy(pose.to_array()).float().to(
            self.device)
        return pc_points, pose_matrix

    def __len__(self):
        return len(self.loader)