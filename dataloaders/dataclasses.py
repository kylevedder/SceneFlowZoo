from dataclasses import dataclass

import numpy as np
import torch
from bucketed_scene_flow_eval.datastructures import (
    TimeSyncedSceneFlowFrame,
    RGBImage,
    RGBFrameLookup,
    EgoLidarFlow,
)
from bucketed_scene_flow_eval.interfaces import (
    NonCausalSeqLoaderDataset,
    CausalSeqLoaderDataset,
    LoaderType,
)

from typing import Type, Union

from pointclouds import (
    from_fixed_array_np,
    from_fixed_array_torch,
    to_fixed_array_np,
    from_fixed_array_valid_mask_torch,
)


@dataclass
class BucketedSceneFlowInputSequence:
    """
    Class that contains all the data required for computing scene flow of a single dataset sample,
    which may contain multiple observations.

    Args:
        dataset_idx (int): The index of the dataset.
        sequence_log_id (str): Unique identifier for the dataset log.
        sequence_idx (int): Index of the sample in the dataset.
        full_pc (torch.Tensor): The full point cloud as a float tensor of shape (K, PadN, 3),
            where K is the number of point clouds, PadN is the padded number of points per point cloud,
            and 3 represents the XYZ coordinates.
        full_pc_mask (torch.Tensor): A boolean mask for the full point cloud of shape (K, PadN,),
            indicating valid points.
        full_pc_gt_flowed (torch.Tensor): The ground truth flowed point cloud as a float tensor
            of shape (K - 1, PadN, 3), following the same conventions as `full_pc`.
        full_pc_gt_flowed_mask (torch.Tensor): A boolean mask for the ground truth flowed point
            cloud of shape (K - 1, PadN, ).
        full_pc_gt_class (torch.Tensor): The ground truth class for each point in the point cloud
            as a long tensor of shape (K - 1, PadN).
        pc_poses_sensor_to_ego (torch.Tensor): Sensor to ego poses for each point cloud as a float tensor
            of shape (K, 4, 4).
        pc_poses_ego_to_global (torch.Tensor): Ego to global poses for each point cloud as a float tensor
            of shape (K, 4, 4).
        rgb_images (torch.Tensor): RGB images as a float tensor of shape (K, NumIm, 4, H, W),
            where K is the number of observations, NumIm is the number of images per observation,
            3 represents the RGB channels, and H, W are the image height and width, respectively.
        rgb_poses_sensor_to_ego (torch.Tensor): Sensor to ego poses for each RGB image as a float tensor
            of shape (K, NumIm, 4, 4).
        rgb_poses_ego_to_global (torch.Tensor): Ego to global poses for each RGB image as a float tensor
            of shape (K, NumIm, 4, 4).
        loader_type (LoaderType): The operation mode of the dataset.
    """

    dataset_idx: int
    sequence_log_id: str
    sequence_idx: int

    # PC Data
    full_pc: torch.Tensor  # (K, PadN, 3)
    full_pc_mask: torch.Tensor  # (K, PadN,)
    full_pc_gt_flowed: torch.Tensor  # (K-1, PadN, 3)
    full_pc_gt_flowed_mask: torch.Tensor  # (K-1, PadN, )
    full_pc_gt_class: torch.Tensor  # (K-1, PadN,)
    pc_poses_sensor_to_ego: torch.Tensor  # (K, 4, 4)
    pc_poses_ego_to_global: torch.Tensor  # (K, 4, 4)

    # RGB Data
    rgb_images: torch.Tensor  # (K, NumIm, 4, H, W)
    rgb_poses_sensor_to_ego: torch.Tensor  # (K, NumIm, 4, 4)
    rgb_poses_ego_to_global: torch.Tensor  # (K, NumIm, 4, 4)

    # Operation Mode
    loader_type: LoaderType

    def get_full_ego_pc(self, idx: int) -> torch.Tensor:
        """
        Get the point cloud at the specified index.
        """
        return from_fixed_array_torch(self.full_pc[idx])

    def get_full_ego_pc_gt_flowed(self, idx: int) -> torch.Tensor:
        """
        Get the point cloud gt flow at the specified index.
        """
        return from_fixed_array_torch(self.full_pc_gt_flowed[idx])

    def get_full_pc_mask(self, idx: int) -> torch.Tensor:
        """
        Get the point cloud mask at the specified index.
        """
        return from_fixed_array_torch(self.full_pc_mask[idx]) > 0

    def get_full_pc_gt_flow_mask(self, idx: int) -> torch.Tensor:
        """
        Get the point cloud gt flow mask at the specified index.
        """
        return from_fixed_array_torch(self.full_pc_gt_flowed_mask[idx]) > 0

    def get_ego_pc(self, idx: int) -> torch.Tensor:
        full_pc = self.get_full_ego_pc(idx)
        full_mask = self.get_full_pc_mask(idx)
        return full_pc[full_mask]

    def get_ego_pc_gt_flowed(self, idx: int) -> torch.Tensor:
        full_pc = self.get_full_ego_pc_gt_flowed(idx)
        full_mask = self.get_full_pc_gt_flow_mask(idx)
        return full_pc[full_mask]

    def get_full_global_pc(self, idx: int) -> torch.Tensor:
        ego_pc = self.get_full_ego_pc(idx)
        sensor_to_ego, ego_to_global = self.get_pc_transform_matrices(idx)
        sensor_to_global = torch.matmul(ego_to_global, sensor_to_ego)
        # Ego PC is Nx3, we need to add another entry to make it Nx4 for the transformation
        ego_pc = torch.cat([ego_pc, torch.ones(ego_pc.shape[0], 1, device=ego_pc.device)], dim=1)
        return torch.matmul(sensor_to_global, ego_pc.T).T[:, :3]

    def get_full_global_pc_gt_flowed(self, idx: int) -> torch.Tensor:
        ego_pc = self.get_full_ego_pc_gt_flowed(idx)
        sensor_to_ego, ego_to_global = self.get_pc_transform_matrices(idx)
        sensor_to_global = torch.matmul(ego_to_global, sensor_to_ego)
        # Ego PC is Nx3, we need to add another entry to make it Nx4 for the transformation
        ego_pc = torch.cat([ego_pc, torch.ones(ego_pc.shape[0], 1, device=ego_pc.device)], dim=1)
        return torch.matmul(sensor_to_global, ego_pc.T).T[:, :3]

    def get_global_pc(self, idx: int) -> torch.Tensor:
        full_pc = self.get_full_global_pc(idx)
        full_mask = self.get_full_pc_mask(idx)
        return full_pc[full_mask]

    def get_global_pc_gt_flowed(self, idx: int) -> torch.Tensor:
        full_pc = self.get_full_global_pc_gt_flowed(idx)
        full_mask = self.get_full_pc_gt_flow_mask(idx)
        return full_pc[full_mask]

    def get_pc_transform_matrices(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pc_poses_sensor_to_ego[idx], self.pc_poses_ego_to_global[idx]

    def __post_init__(self):
        # Check shapes for point cloud data
        assert (
            self.full_pc.dim() == 3 and self.full_pc.shape[2] == 3
        ), f"Expected full_pc to have shape (K, PadN, 3), got {self.full_pc.shape}"
        assert (
            self.full_pc_mask.dim() == 2
        ), f"Expected full_pc_mask to have shape (K, PadN), got {self.full_pc_mask.shape}"
        assert (
            self.full_pc_mask.shape == self.full_pc.shape[:2]
        ), f"Expected full_pc_mask to match the first two dimensions of full_pc {self.full_pc.shape[:2]}, got {self.full_pc_mask.shape}"
        assert (
            self.full_pc_gt_flowed.dim() == 3
            and self.full_pc_gt_flowed.shape[2] == 3
            and self.full_pc_gt_flowed.shape[0] == self.full_pc.shape[0] - 1
        ), f"Expected full_pc_gt_flowed to have shape (K-1, PadN, 3), got {self.full_pc_gt_flowed.shape}"
        assert (
            self.full_pc_gt_flowed_mask.dim() == 2
            and self.full_pc_gt_flowed_mask.shape == self.full_pc_gt_flowed.shape[:2]
        ), f"Expected full_pc_gt_flowed_mask to have shape (K-1, PadN), matching the first two dimensions of full_pc_gt_flowed {self.full_pc_gt_flowed.shape[:2]}, got {self.full_pc_gt_flowed_mask.shape}"
        assert (
            self.full_pc_gt_class.dim() == 2
            and self.full_pc_gt_class.shape == self.full_pc_gt_flowed.shape[:2]
        ), f"Expected full_pc_gt_class to have shape (K-1, PadN), matching the first two dimensions of full_pc_gt_flowed {self.full_pc_gt_flowed.shape[:2]}, got {self.full_pc_gt_class.shape}"

        # Check shapes for poses
        assert self.pc_poses_sensor_to_ego.dim() == 3 and self.pc_poses_sensor_to_ego.shape[1:] == (
            4,
            4,
        ), f"Expected pc_poses_sensor_to_ego to have shape (K, 4, 4), got {self.pc_poses_sensor_to_ego.shape}"
        assert (
            self.pc_poses_ego_to_global.dim() == 3
            and self.pc_poses_ego_to_global.shape == self.pc_poses_sensor_to_ego.shape
        ), f"Expected pc_poses_ego_to_global to have the same shape as pc_poses_sensor_to_ego {self.pc_poses_sensor_to_ego.shape}, got {self.pc_poses_ego_to_global.shape}"

        # Check shapes for RGB data
        assert (
            self.rgb_images.dim() == 5 and self.rgb_images.shape[2] == 4
        ), f"Expected rgb_images to have shape (K, NumIm, 4, H, W), got {self.rgb_images.shape}"
        assert (
            self.rgb_poses_sensor_to_ego.dim() == 4
            and self.rgb_poses_sensor_to_ego.shape[0] == self.rgb_images.shape[0]
            and self.rgb_poses_sensor_to_ego.shape[1] == self.rgb_images.shape[1]
            and self.rgb_poses_sensor_to_ego.shape[2:] == (4, 4)
        ), f"Expected rgb_poses_sensor_to_ego to have shape (K, NumIm, 4, 4), matching the first two dimensions of rgb_images {self.rgb_images.shape[:2]}, got {self.rgb_poses_sensor_to_ego.shape}"
        assert (
            self.rgb_poses_ego_to_global.dim() == 4
            and self.rgb_poses_ego_to_global.shape == self.rgb_poses_sensor_to_ego.shape
        ), f"Expected rgb_poses_ego_to_global to have the same shape as rgb_poses_sensor_to_ego {self.rgb_poses_sensor_to_ego.shape}, got {self.rgb_poses_ego_to_global.shape}"

        assert isinstance(
            self.loader_type, LoaderType
        ), f"Expected operation_mode to be a LoaderType, got {type(self.loader_type)}"

    def __len__(self) -> int:
        return self.full_pc.shape[0]

    @staticmethod
    def from_frame_list(
        idx: int,
        frame_list: list[TimeSyncedSceneFlowFrame],
        pc_max_len: int,
        loader_type: LoaderType,
    ) -> "BucketedSceneFlowInputSequence":
        """
        Create a BucketedSceneFlowItem from a list of TimeSyncedSceneFlowFrame objects.
        """

        dataset_log_id = frame_list[0].log_id
        dataset_idx = frame_list[0].log_idx

        # PC data
        full_pc = torch.stack(
            [
                torch.from_numpy(to_fixed_array_np(frame.pc.full_pc.points, max_len=pc_max_len))
                for frame in frame_list
            ]
        )
        full_pc_mask = torch.stack(
            [
                torch.from_numpy(to_fixed_array_np(frame.pc.mask, max_len=pc_max_len))
                for frame in frame_list
            ]
        )

        flow_frame_list = frame_list[:-1]
        flowed_pc_frame_list = [frame.pc.flow(frame.flow) for frame in flow_frame_list]
        assert (
            len(flowed_pc_frame_list) == len(frame_list) - 1
        ), f"Expected {len(frame_list) - 1} flowed point clouds, got {len(flowed_pc_frame_list)}"

        full_pc_gt_flowed = torch.stack(
            [
                torch.from_numpy(to_fixed_array_np(frame.full_pc.points, max_len=pc_max_len))
                for frame in flowed_pc_frame_list
            ]
        )

        full_pc_gt_flowed_mask = torch.stack(
            [
                torch.from_numpy(to_fixed_array_np(frame.mask, max_len=pc_max_len))
                for frame in flowed_pc_frame_list
            ]
        )

        full_pc_gt_class = torch.stack(
            [
                torch.from_numpy(to_fixed_array_np(frame.pc.full_pc_classes, max_len=pc_max_len))
                for frame in flow_frame_list
            ]
        )

        pc_poses_sensor_to_ego = torch.stack(
            [torch.from_numpy(frame.pc.pose.sensor_to_ego.to_array()) for frame in frame_list]
        )
        pc_poses_ego_to_global = torch.stack(
            [torch.from_numpy(frame.pc.pose.ego_to_global.to_array()) for frame in frame_list]
        )

        def _concatenate_image_channels(rgb_image: RGBImage) -> torch.Tensor:
            full_img = rgb_image.full_image
            is_valid_mask = np.expand_dims(rgb_image.get_is_valid_mask(), axis=2)
            img = torch.from_numpy(np.concatenate([full_img, is_valid_mask], axis=2))
            assert img.shape == (
                full_img.shape[0],
                full_img.shape[1],
                4,
            ), f"Invalid image shape {img.shape}"
            return img

        def _make_image_stack(image_tensors: list[torch.Tensor]) -> torch.Tensor:
            if len(image_tensors) == 0:
                return torch.zeros(0, 4, 0, 0)
            img_stack = torch.stack(image_tensors)
            assert img_stack.shape[0] == len(image_tensors), (
                f"Invalid image stack shape {img_stack.shape}, "
                f"expected {len(image_tensors)} images, got {img_stack.shape[0]}"
            )
            # Convert from (NumIm, H, W, 4) to (NumIm, 4, H, W)
            return img_stack.permute(0, 3, 1, 2)

        # RGB data
        rgb_images = torch.stack(
            [
                _make_image_stack(
                    [_concatenate_image_channels(img.rgb) for img in frame.rgbs.values()]
                )
                for frame in frame_list
            ]
        )

        def _concatenate_transforms(transforms: list[np.ndarray]) -> torch.Tensor:
            if len(transforms) == 0:
                return torch.zeros(0, 4, 4)
            return torch.stack([torch.from_numpy(transform) for transform in transforms])

        rgb_poses_sensor_to_ego = torch.stack(
            [
                _concatenate_transforms(
                    [img.pose.sensor_to_ego.to_array() for img in frame.rgbs.values()]
                )
                for frame in frame_list
            ]
        )
        rgb_poses_ego_to_global = torch.stack(
            [
                _concatenate_transforms(
                    [img.pose.ego_to_global.to_array() for img in frame.rgbs.values()]
                )
                for frame in frame_list
            ]
        )

        return BucketedSceneFlowInputSequence(
            dataset_idx=idx,
            sequence_log_id=dataset_log_id,
            sequence_idx=dataset_idx,
            full_pc=full_pc.float(),
            full_pc_mask=full_pc_mask.float(),
            full_pc_gt_flowed=full_pc_gt_flowed.float(),
            full_pc_gt_flowed_mask=full_pc_gt_flowed_mask.float(),
            full_pc_gt_class=full_pc_gt_class.float(),
            pc_poses_sensor_to_ego=pc_poses_sensor_to_ego.float(),
            pc_poses_ego_to_global=pc_poses_ego_to_global.float(),
            rgb_images=rgb_images.float(),
            rgb_poses_sensor_to_ego=rgb_poses_sensor_to_ego.float(),
            rgb_poses_ego_to_global=rgb_poses_ego_to_global.float(),
            loader_type=loader_type,
        )

    def to(self, device: str | torch.device) -> "BucketedSceneFlowInputSequence":
        """
        Copy tensors in this batch to the target device.

        Args:
            device: the string (and optional ordinal) used to construct the device object, e.g., 'cuda:0'
        """
        # Update tensors to the new device
        self.full_pc = self.full_pc.to(device)
        self.full_pc_mask = self.full_pc_mask.to(device)
        self.full_pc_gt_flowed = self.full_pc_gt_flowed.to(device)
        self.full_pc_gt_flowed_mask = self.full_pc_gt_flowed_mask.to(device)
        self.full_pc_gt_class = self.full_pc_gt_class.to(device)
        self.pc_poses_sensor_to_ego = self.pc_poses_sensor_to_ego.to(device)
        self.pc_poses_ego_to_global = self.pc_poses_ego_to_global.to(device)
        self.rgb_images = self.rgb_images.to(device)
        self.rgb_poses_sensor_to_ego = self.rgb_poses_sensor_to_ego.to(device)
        self.rgb_poses_ego_to_global = self.rgb_poses_ego_to_global.to(device)

        return self

    def clone(self) -> "BucketedSceneFlowInputSequence":
        """
        Clone this object.
        """
        return BucketedSceneFlowInputSequence(
            dataset_idx=self.dataset_idx,
            sequence_log_id=self.sequence_log_id,
            sequence_idx=self.sequence_idx,
            full_pc=self.full_pc.clone(),
            full_pc_mask=self.full_pc_mask.clone(),
            full_pc_gt_flowed=self.full_pc_gt_flowed.clone(),
            full_pc_gt_flowed_mask=self.full_pc_gt_flowed_mask.clone(),
            full_pc_gt_class=self.full_pc_gt_class.clone(),
            pc_poses_sensor_to_ego=self.pc_poses_sensor_to_ego.clone(),
            pc_poses_ego_to_global=self.pc_poses_ego_to_global.clone(),
            rgb_images=self.rgb_images.clone(),
            rgb_poses_sensor_to_ego=self.rgb_poses_sensor_to_ego.clone(),
            rgb_poses_ego_to_global=self.rgb_poses_ego_to_global.clone(),
            loader_type=self.loader_type,
        )

    def detach(self) -> "BucketedSceneFlowInputSequence":
        """
        Detach all tensors in this object.
        """
        self.full_pc = self.full_pc.detach()
        self.full_pc_mask = self.full_pc_mask.detach()
        self.full_pc_gt_flowed = self.full_pc_gt_flowed.detach()
        self.full_pc_gt_flowed_mask = self.full_pc_gt_flowed_mask.detach()
        self.full_pc_gt_class = self.full_pc_gt_class.detach()
        self.pc_poses_sensor_to_ego = self.pc_poses_sensor_to_ego.detach()
        self.pc_poses_ego_to_global = self.pc_poses_ego_to_global.detach()
        self.rgb_images = self.rgb_images.detach()
        self.rgb_poses_sensor_to_ego = self.rgb_poses_sensor_to_ego.detach()
        self.rgb_poses_ego_to_global = self.rgb_poses_ego_to_global.detach()
        return self

    def requires_grad_(self, requires_grad: bool) -> "BucketedSceneFlowInputSequence":
        """
        Set the requires_grad attribute of all tensors in this object.
        """
        self.full_pc.requires_grad_(requires_grad)
        self.full_pc_mask.requires_grad_(requires_grad)
        self.full_pc_gt_flowed.requires_grad_(requires_grad)
        self.full_pc_gt_flowed_mask.requires_grad_(requires_grad)
        self.full_pc_gt_class.requires_grad_(requires_grad)
        self.pc_poses_sensor_to_ego.requires_grad_(requires_grad)
        self.pc_poses_ego_to_global.requires_grad_(requires_grad)
        self.rgb_images.requires_grad_(requires_grad)
        self.rgb_poses_sensor_to_ego.requires_grad_(requires_grad)
        self.rgb_poses_ego_to_global.requires_grad_(requires_grad)
        return self

    def slice(self, start_idx: int, end_idx: int) -> "BucketedSceneFlowInputSequence":
        # Slice the tensors in this object
        # For K length tensors, the slice is [start_idx:end_idx]
        # For K - 1 length tensors, the slice is [start_idx:end_idx - 1]

        return BucketedSceneFlowInputSequence(
            dataset_idx=self.dataset_idx,
            sequence_log_id=self.sequence_log_id,
            sequence_idx=self.sequence_idx,
            full_pc=self.full_pc[start_idx:end_idx],
            full_pc_mask=self.full_pc_mask[start_idx:end_idx],
            full_pc_gt_flowed=self.full_pc_gt_flowed[start_idx : end_idx - 1],
            full_pc_gt_flowed_mask=self.full_pc_gt_flowed_mask[start_idx : end_idx - 1],
            full_pc_gt_class=self.full_pc_gt_class[start_idx : end_idx - 1],
            pc_poses_sensor_to_ego=self.pc_poses_sensor_to_ego[start_idx:end_idx],
            pc_poses_ego_to_global=self.pc_poses_ego_to_global[start_idx:end_idx],
            rgb_images=self.rgb_images[start_idx:end_idx],
            rgb_poses_sensor_to_ego=self.rgb_poses_sensor_to_ego[start_idx:end_idx],
            rgb_poses_ego_to_global=self.rgb_poses_ego_to_global[start_idx:end_idx],
            loader_type=self.loader_type,
        )

    def reverse(self) -> "BucketedSceneFlowInputSequence":
        # Reverse the first dimension of all tensors in this object
        return BucketedSceneFlowInputSequence(
            dataset_idx=self.dataset_idx,
            sequence_log_id=self.sequence_log_id,
            sequence_idx=self.sequence_idx,
            full_pc=self.full_pc.flip(0),
            full_pc_mask=self.full_pc_mask.flip(0),
            full_pc_gt_flowed=self.full_pc_gt_flowed.flip(0),
            full_pc_gt_flowed_mask=self.full_pc_gt_flowed_mask.flip(0),
            full_pc_gt_class=self.full_pc_gt_class.flip(0),
            pc_poses_sensor_to_ego=self.pc_poses_sensor_to_ego.flip(0),
            pc_poses_ego_to_global=self.pc_poses_ego_to_global.flip(0),
            rgb_images=self.rgb_images.flip(0),
            rgb_poses_sensor_to_ego=self.rgb_poses_sensor_to_ego.flip(0),
            rgb_poses_ego_to_global=self.rgb_poses_ego_to_global.flip(0),
            loader_type=self.loader_type,
        )

    @property
    def device(self) -> torch.device:
        """
        Get the device of the tensors in this object.
        """
        return self.full_pc.device


@dataclass
class BucketedSceneFlowOutputSequence:
    """
    A standardized set of outputs for Bucketed Scene Flow evaluation.

    Args:
        ego_flows: torch.Tensor  # (K - 1, PadN, 3)
        valid_flow_mask: torch.Tensor  # (K - 1, PadN,)
    """

    ego_flows: torch.Tensor  # (K - 1, PadN, 3)
    valid_flow_mask: torch.Tensor  # (K - 1, PadN, )

    @staticmethod
    def from_ego_lidar_flow_list(
        ego_lidar_flows: list[EgoLidarFlow], max_len: int
    ) -> "BucketedSceneFlowOutputSequence":
        """
        Create a BucketedSceneFlowOutputSequence from a list of EgoLidarFlow objects.
        """

        ego_flows = torch.stack(
            [
                torch.from_numpy(to_fixed_array_np(flow.full_flow, max_len=max_len))
                for flow in ego_lidar_flows
            ]
        )
        valid_flow_mask = torch.stack(
            [
                torch.from_numpy(to_fixed_array_np(flow.mask, max_len=max_len))
                for flow in ego_lidar_flows
            ]
        )

        return BucketedSceneFlowOutputSequence(
            ego_flows=ego_flows.float(), valid_flow_mask=valid_flow_mask.float()
        )

    def get_full_ego_flow(self, idx: int) -> torch.Tensor:
        """
        Get the ego flow at the specified index.
        """
        return from_fixed_array_torch(self.ego_flows[idx])

    def get_full_flow_mask(self, idx: int) -> torch.Tensor:
        """
        Get the flow mask at the specified index.
        """
        return from_fixed_array_torch(self.valid_flow_mask[idx]) > 0

    def __post_init__(self):
        # Check the shape of the ego flows
        assert (
            self.ego_flows.dim() == 3 and self.ego_flows.shape[-1] == 3
        ), f"Expected ego_flows to have shape (K - 1, PadN, 3), got {self.ego_flows.shape}"

        # Check the shape of the valid flow mask
        assert (
            self.valid_flow_mask.dim() == 2
        ), f"Expected valid_flow_mask to have shape (K - 1, PadN), got {self.valid_flow_mask.shape}"
        assert (
            self.valid_flow_mask.shape == self.ego_flows.shape[:2]
        ), f"Shape mismatch: ego_flows {self.ego_flows.shape[:2]} vs valid_flow_mask {self.valid_flow_mask.shape}"

        # Check that the ego flows and valid flow mask have the same device
        assert (
            self.ego_flows.device == self.valid_flow_mask.device
        ), f"Device mismatch: {self.ego_flows.device} vs {self.valid_flow_mask.device}"

    def __len__(self) -> int:
        return self.ego_flows.shape[0]

    def to(self, device: str | torch.device) -> "BucketedSceneFlowOutputSequence":
        """
        Copy tensors in this batch to the target device.

        Args:
            device: the string (and optional ordinal) used to construct the device object ex. 'cuda:0'
        """
        self.ego_flows = self.ego_flows.to(device)
        self.valid_flow_mask = self.valid_flow_mask.to(device)

        return self

    def to_ego_lidar_flow_list(self) -> list[EgoLidarFlow]:
        """
        Convert the ego flows and valid flow mask to a list of EgoLidarFlow objects.
        """

        def _to_ego_lidar_flow(
            padded_flow: torch.Tensor, padded_mask: torch.Tensor
        ) -> EgoLidarFlow:
            unpad = from_fixed_array_valid_mask_torch(padded_mask)
            return EgoLidarFlow(
                full_flow=padded_flow[unpad].detach().cpu().numpy(),
                mask=(padded_mask[unpad] != 0).detach().cpu().numpy(),
            )

        return [
            _to_ego_lidar_flow(flow, mask)
            for flow, mask in zip(self.ego_flows, self.valid_flow_mask)
        ]

    def reverse(self) -> "BucketedSceneFlowOutputSequence":
        # Reverse the first dimension of all tensors in this object, and reverse the actual flow direction
        return BucketedSceneFlowOutputSequence(
            ego_flows=-self.ego_flows.flip(0),
            valid_flow_mask=self.valid_flow_mask.flip(0),
        )

    def clone(self) -> "BucketedSceneFlowOutputSequence":
        """
        Clone this object.
        """
        return BucketedSceneFlowOutputSequence(
            ego_flows=self.ego_flows.clone(),
            valid_flow_mask=self.valid_flow_mask.clone(),
        )

    def detach(self) -> "BucketedSceneFlowOutputSequence":
        """
        Detach all tensors in this object.
        """
        self.ego_flows = self.ego_flows.detach()
        self.valid_flow_mask = self.valid_flow_mask.detach()
        return self

    def requires_grad_(self, requires_grad: bool) -> "BucketedSceneFlowOutputSequence":
        """
        Set the requires_grad attribute of all tensors in this object.
        """
        self.ego_flows.requires_grad_(requires_grad)
        self.valid_flow_mask.requires_grad_(requires_grad)
        return self
