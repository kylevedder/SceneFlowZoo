import torch
import torch.nn as nn
from pytorch3d.ops.knn import knn_points

from model.embedders import Embedder
from model.backbones import FeaturePyramidNetwork
from model.attention import JointConvAttention
from model.heads import NeuralSceneFlowPrior

from typing import List, Tuple
from pointclouds import PointCloud, SE3


def _torch_to_jagged_pc(torch_tensor: torch.Tensor) -> PointCloud:
    assert isinstance(
        torch_tensor,
        torch.Tensor), f"Expected torch.Tensor, got {type(torch_tensor)}"
    return PointCloud.from_fixed_array(torch_tensor)


def _pc_to_torch(pc: PointCloud, device) -> torch.Tensor:
    assert isinstance(pc, PointCloud), f"Expected PointCloud, got {type(pc)}"
    return pc.points.float().to(device)


class JointFlowLoss():

    def __init__(self,
                 device: str,
                 NSFP_FILTER_SIZE=64,
                 NSFP_NUM_LAYERS=4) -> None:
        self.device = device
        self.nsfp = NeuralSceneFlowPrior(num_hidden_units=NSFP_FILTER_SIZE,
                                         num_hidden_layers=NSFP_NUM_LAYERS).to(self.device)

    def __call__(self,
                 batched_sequence: List[List[Tuple[torch.Tensor, torch.Tensor,
                                                   torch.Tensor]]],
                 delta: int) -> torch.Tensor:
        loss = 0
        for sequence in batched_sequence:
            sequence_length = len(sequence)
            assert delta > 0, f"delta must be positive, got {delta}"
            assert delta < sequence_length, f"delta must be less than sequence_length, got {delta}"
            for i in range(sequence_length - delta):
                pc_t0, _, _ = sequence[i]
                pc_t1, _, _ = sequence[i + delta]
                pc_t0 = _torch_to_jagged_pc(pc_t0)
                pc_t1 = _torch_to_jagged_pc(pc_t1)
                pc_t0 = _pc_to_torch(pc_t0, self.device)
                pc_t1 = _pc_to_torch(pc_t1, self.device)

                pc_t0_warped_to_t1 = pc_t0
                param_list = []
                for j in range(delta):
                    _, _, nsfp_params = sequence[i + j]
                    pc_t0_warped_to_t1 = self.nsfp(pc_t0_warped_to_t1,
                                                   nsfp_params)
                    param_list.append(nsfp_params)
                loss += self._warped_pc_loss(pc_t0_warped_to_t1, pc_t1,
                                             param_list)
        return loss

    def _warped_pc_loss(self,
                        pc_t0_warped_to_t1: torch.Tensor,
                        pc_t1: torch.Tensor,
                        nsfp_param_list: List[torch.Tensor],
                        dist_threshold=2.0,
                        param_regularizer=10e-4):
        loss = 0
        batched_warped_pc_t = pc_t0_warped_to_t1.unsqueeze(0)
        batched_pc_t1 = pc_t1.unsqueeze(0)

        # Compute min distance between warped point cloud and point cloud at t+1.

        warped_pc_t_shape_tensor = torch.LongTensor(
            [pc_t0_warped_to_t1.shape[0]]).to(batched_warped_pc_t.device)
        pc_t1_shape_tensor = torch.LongTensor([pc_t1.shape[0]
                                               ]).to(batched_pc_t1.device)
        warped_to_t1_knn = knn_points(p1=batched_warped_pc_t,
                                      p2=batched_pc_t1,
                                      lengths1=warped_pc_t_shape_tensor,
                                      lengths2=pc_t1_shape_tensor,
                                      K=1)
        warped_to_t1_distances = warped_to_t1_knn.dists[0]
        t1_to_warped_knn = knn_points(p1=batched_pc_t1,
                                      p2=batched_warped_pc_t,
                                      lengths1=pc_t1_shape_tensor,
                                      lengths2=warped_pc_t_shape_tensor,
                                      K=1)
        t1_to_warped_distances = t1_to_warped_knn.dists[0]
        # breakpoint()

        # Throw out distances that are too large (beyond the dist threshold).
        # warped_to_t1_distances[warped_to_t1_distances > dist_threshold] = 0
        # t1_to_warped_distances[t1_to_warped_distances > dist_threshold] = 0

        # Add up the distances.
        loss += torch.sum(warped_to_t1_distances) + torch.sum(
            t1_to_warped_distances)

        # L2 regularization on the neural scene flow prior parameters.
        for nsfp_params in nsfp_param_list:
            loss += torch.sum(nsfp_params**2 * param_regularizer)

        return loss


class JointFlow(nn.Module):

    def __init__(self,
                 batch_size: int,
                 device: str,
                 VOXEL_SIZE=(0.14, 0.14, 4),
                 PSEUDO_IMAGE_DIMS=(512, 512),
                 POINT_CLOUD_RANGE=(-33.28, -33.28, -3, 33.28, 33.28, 1),
                 MAX_POINTS_PER_VOXEL=128,
                 FEATURE_CHANNELS=16,
                 FILTERS_PER_BLOCK=3,
                 PYRAMID_LAYERS=1,
                 SEQUENCE_LENGTH=5,
                 NSFP_FILTER_SIZE=64,
                 NSFP_NUM_LAYERS=4) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.embedder = Embedder(voxel_size=VOXEL_SIZE,
                                 pseudo_image_dims=PSEUDO_IMAGE_DIMS,
                                 point_cloud_range=POINT_CLOUD_RANGE,
                                 max_points_per_voxel=MAX_POINTS_PER_VOXEL,
                                 feat_channels=FEATURE_CHANNELS)

        self.pyramid = FeaturePyramidNetwork(
            pseudoimage_dims=PSEUDO_IMAGE_DIMS,
            input_num_channels=FEATURE_CHANNELS + 2,
            num_filters_per_block=FILTERS_PER_BLOCK,
            num_layers_of_pyramid=PYRAMID_LAYERS)

        self.nsfp = NeuralSceneFlowPrior(num_hidden_units=NSFP_FILTER_SIZE,
                                         num_hidden_layers=NSFP_NUM_LAYERS)
        self.attention = JointConvAttention(
            pseudoimage_dims=PSEUDO_IMAGE_DIMS,
            sequence_length=SEQUENCE_LENGTH,
            per_element_num_channels=(FEATURE_CHANNELS + 2) *
            (PYRAMID_LAYERS + 1),
            per_element_output_params=self.nsfp.param_count)
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH
        self.PSEUDO_IMAGE_DIMS = PSEUDO_IMAGE_DIMS

    def forward(
        self, batched_sequence: List[Tuple[PointCloud, SE3]]
    ) -> List[Tuple[PointCloud, SE3, torch.Tensor]]:
        assert len(
            batched_sequence
        ) == self.SEQUENCE_LENGTH, f"Expected sequence of length {self.SEQUENCE_LENGTH}, got {len(batched_sequence)}"

        batched_results = []
        for batch_idx in range(self.batch_size):
            sequence = [(e['pc'][batch_idx], e['pose'][batch_idx])
                        for e in batched_sequence]

            pc_embedding_list = [
                self._embed_pc(pc, pose) for pc, pose in sequence
            ]
            pc_embedding_stack = torch.cat(pc_embedding_list, dim=1)
            nsfp_weights_tensor = torch.squeeze(
                self.attention(pc_embedding_stack), dim=0)
            nsfp_weights_list = torch.split(nsfp_weights_tensor,
                                            self.nsfp.param_count,
                                            dim=0)
            res = [(pc, pose, nsfp_weights)
                   for (pc,
                        pose), nsfp_weights in zip(sequence, nsfp_weights_list)
                   ]
            batched_results.append(res)
        return batched_results

    def _get_device(self):
        return self.device

    def _embed_pc(self, pc: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            pc, torch.Tensor), f"Expected torch.Tensor, got {type(pc)}"
        assert isinstance(
            pose, torch.Tensor), f"Expected torch.Tensor, got {type(pose)}"
        # The pc is in a global frame relative to the initial pose of the sequence.
        # The pose is the pose of the current point cloud relative to the initial pose of the sequence.
        # In order to voxelize the point cloud, we need to translate it to the origin.
        pc_pointcloud = PointCloud.from_fixed_array(pc)
        pose_se3 = SE3.from_array(pose)
        translated_pc = pc_pointcloud.translate(-pose_se3.translation)
        translated_pc_torch = _pc_to_torch(translated_pc, self._get_device())

        translation_x_torch = torch.ones(
            (1, 1, self.PSEUDO_IMAGE_DIMS[0], self.PSEUDO_IMAGE_DIMS[1])).to(
                self._get_device()) * pose_se3.translation[0]
        translation_y_torch = torch.ones(
            (1, 1, self.PSEUDO_IMAGE_DIMS[0], self.PSEUDO_IMAGE_DIMS[1])).to(
                self._get_device()) * pose_se3.translation[1]

        pseudoimage = self.embedder(translated_pc_torch)
        pseudoimage = torch.cat(
            (pseudoimage, translation_x_torch, translation_y_torch), dim=1)
        latent = self.pyramid(pseudoimage)
        return latent
