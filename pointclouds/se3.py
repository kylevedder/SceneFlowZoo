import numpy as np
from pyquaternion import Quaternion


class SE3:
    """An SE3 class allows point cloud rotation and translation operations."""

    def __init__(self, rotation_matrix: np.ndarray,
                 translation: np.ndarray) -> None:
        """Initialize an SE3 instance with its rotation and translation matrices.
        Args:
            rotation: Array of shape (3, 3)
            translation: Array of shape (3,)
        """
        assert rotation_matrix.shape == (3, 3)
        assert translation.shape == (3, )
        self.rotation_matrix = rotation_matrix
        self.translation = translation

        if isinstance(self.rotation_matrix, np.ndarray):
            self.transform_matrix = np.eye(4)
        else:
            import torch
            self.transform_matrix = torch.eye(4)
        self.transform_matrix[:3, :3] = self.rotation_matrix
        self.transform_matrix[:3, 3] = self.translation

    @staticmethod
    def from_rot_w_x_y_z_translation_x_y_z(rw, rx, ry, rz, tx, ty, tz) -> None:
        rotation_matrix = Quaternion(w=rw, x=rx, y=ry, z=rz).rotation_matrix
        translation = np.array([tx, ty, tz])
        return SE3(rotation_matrix, translation)

    def transform_points(self, point_cloud: np.ndarray) -> np.ndarray:
        """Apply the SE(3) transformation to this point cloud.
        Args:
            point_cloud: Array of shape (N, 3). If the transform represents dst_SE3_src,
                then point_cloud should consist of points in frame `src`
        Returns:
            Array of shape (N, 3) representing the transformed point cloud, i.e. points in frame `dst`
        """
        return point_cloud @ self.rotation_matrix.T + self.translation

    def inverse_transform_points(self, point_cloud: np.ndarray) -> np.ndarray:
        """Undo the translation and then the rotation (Inverse SE(3) transformation)."""
        return (point_cloud.copy() - self.translation) @ self.rotation_matrix

    def inverse(self) -> "SE3":
        """Return the inverse of the current SE3 transformation.
        For example, if the current object represents target_SE3_src, we will return instead src_SE3_target.
        Returns:
            src_SE3_target: instance of SE3 class, representing
                inverse of SE3 transformation target_SE3_src
        """
        return SE3(rotation_matrix=self.rotation_matrix.T,
                   translation=self.rotation_matrix.T.dot(-self.translation))

    def compose(self, right_se3: "SE3") -> "SE3":
        """Compose (right multiply) this class' transformation matrix T with another SE3 instance.
        Algebraic representation: chained_se3 = T * right_se3
        Args:
            right_se3: another instance of SE3 class
        Returns:
            chained_se3: new instance of SE3 class
        """
        chained_transform_matrix = self.transform_matrix @ right_se3.transform_matrix
        chained_se3 = SE3(
            rotation_matrix=chained_transform_matrix[:3, :3],
            translation=chained_transform_matrix[:3, 3],
        )
        return chained_se3

    def to_array(self) -> np.ndarray:
        """Return the SE3 transformation matrix as a numpy array."""
        return self.transform_matrix

    @staticmethod
    def from_array(transform_matrix: np.ndarray) -> "SE3":
        """Initialize an SE3 instance from a numpy array."""
        return SE3(
            rotation_matrix=transform_matrix[:3, :3],
            translation=transform_matrix[:3, 3],
        )

    def __repr__(self) -> str:
        return "SE3 transform"