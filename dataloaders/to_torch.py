import torch
import numpy as np

class ToTorchDataset(torch.utils.data.Dataset):

    def __init__(self, loader, max_points: int):
        self.loader = loader
        assert max_points > 0, f"max_points must be > 0, got {max_points}"
        self.max_points = max_points

    def __getitem__(self, index: int):
        pc, pose = self.loader[index]
        translated_pc = pc.translate(-pose.translation)

        translated_points = translated_pc.to_array()
        if len(translated_points) > self.max_points:
            translated_points = np.RandomState(0).shuffle(translated_points)[:self.max_points]
        
        # Setup an indicator variable for the points being padded.
        padded_points = np.zeros((self.max_points, translated_points.shape[1] + 1))
        padded_points[:translated_points.shape[0], :translated_points.shape[1]] = translated_points
        padded_points[translated_points.shape[0]:, -1] = 1


        pc_points = torch.from_numpy(padded_points).float()
        pose_matrix = torch.from_numpy(pose.to_array()).float()
        return pc_points, pose_matrix

    def __len__(self):
        return len(self.loader)