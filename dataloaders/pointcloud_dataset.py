import torch
import torch.nn as nn
import numpy as np
# Import rotation as R
from scipy.spatial.transform import Rotation as R


class PointCloudDataset(torch.utils.data.Dataset):

    def __init__(self, sequence_loader, max_pc_points: int = 90000):
        self.sequence_loader = sequence_loader
        self.max_pc_points = max_pc_points

        self.pcs_per_sequence = 295
        self.sequence_ids = sequence_loader.get_sequence_ids()
        self.random_state = np.random.RandomState(len(self.sequence_ids))

    def __len__(self):
        return len(self.sequence_ids) * self.pcs_per_sequence

    def _random_rot(self):
        # Uniform random between -pi and pi
        random_angle = self.random_state.uniform(-np.pi, np.pi)
        # Create random rotation matrix for PC
        return R.from_euler('z', random_angle,
                            degrees=False).as_matrix()[:2, :2]

    def _random_sheer(self):

        # Uniform random scale between 0.95 and 1.05
        random_scale = self.random_state.uniform(0.95, 1.05)
        # Binary random to select upper or lower off diagonal for sheer
        is_upper = self.random_state.randint(2, dtype=bool)

        # Create shear amount between -0.6 and 0.6
        shear_amount = self.random_state.uniform(-0.6, 0.6)

        # Create random sheer matrix for PC
        shear_matrix = np.eye(2) * random_scale
        if is_upper:
            shear_matrix[0, 1] = shear_amount
        else:
            shear_matrix[1, 0] = shear_amount
        return shear_matrix, random_scale, shear_amount, is_upper

    def _random_translation(self):
        # Uniform random scale between -100 and 100 for x, y, -20 and 20 for z
        return self.random_state.uniform([-100, -100, -20], [100, 100, 20])

    def __getitem__(self, index):
        sequence_id = self.sequence_ids[index // self.pcs_per_sequence]
        sequence = self.sequence_loader.load_sequence(sequence_id)
        sequence_idx = index % self.pcs_per_sequence
        pc, _ = sequence.load(sequence_idx, sequence_idx)

        # Create random rotation matrix for PC
        rotation_matrix = self._random_rot()
        pc.points[:2] = rotation_matrix @ pc.points[:2]
        shear_matrix, random_scale, shear_amount, is_upper = self._random_sheer(
        )
        translation = self._random_translation()
        pc.points[:2] = shear_matrix @ pc.points[:2]
        return {
            "pc": pc.to_fixed_array(self.max_pc_points),
            "translation": translation,
            "random_scale": random_scale,
            "shear_amount": shear_amount,
            "is_upper": np.int64(is_upper),
        }
