import torch
import torch.nn as nn
from typing import List, Tuple


class FastFlowDecoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(nn.Linear(128, 32), nn.Linear(32, 3))

    def forward_single(self, before_pseudoimage: torch.Tensor,
                       after_pseudoimage: torch.Tensor, points: torch.Tensor,
                       voxel_coords: torch.Tensor) -> torch.Tensor:
        assert (voxel_coords[:, 0] == 0).all(), "Z index must be 0"

        after_voxel_vectors = after_pseudoimage[:, voxel_coords[:, 1].long(),
                                                voxel_coords[:, 2].long()].T
        before_voxel_vectors = before_pseudoimage[:, voxel_coords[:, 1].long(),
                                                  voxel_coords[:, 2].long()].T
        concatenated_vectors = torch.cat(
            [before_voxel_vectors, after_voxel_vectors], dim=1)

        flow = self.decoder(concatenated_vectors)
        return flow

    def forward(
        self, before_pseudoimages: torch.Tensor,
        after_pseudoimages: torch.Tensor,
        points_coordinates: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> List[torch.Tensor]:

        flow_results = []
        for before_pseudoimage, after_pseudoimage, (points,
                                                    voxel_coords) in zip(
                                                        before_pseudoimages,
                                                        after_pseudoimages,
                                                        points_coordinates):
            flow = self.forward_single(before_pseudoimage, after_pseudoimage,
                                       points, voxel_coords)
            flow_results.append(flow)
        return flow_results
