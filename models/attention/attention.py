import torch
import torch.nn as nn
import numpy as np


class ConvWithNonLinearity(nn.Module):

    def __init__(self, in_num_channels: int, out_num_channels: int,
                 kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.Conv2d(in_num_channels, out_num_channels, kernel_size,
                              stride, padding)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_res = self.conv(x)
        return self.gelu(conv_res)


class JointConvAttention(nn.Module):

    def __init__(self,
                 pseudoimage_dims: tuple,
                 sequence_length: int,
                 per_element_num_channels: int,
                 per_element_output_params: int,
                 latent_image_area_power: int = 2):
        super().__init__()
        assert len(
            pseudoimage_dims
        ) == 2, f"pseudoimage_dims must be a tuple of length 2, got {len(pseudoimage_dims)}"
        assert sequence_length > 0, f"sequence_length must be > 0, got {sequence_length}"
        assert per_element_num_channels > 0, f"per_element_num_channels must be > 0, got {per_element_num_channels}"
        assert per_element_output_params > 0, f"per_element_output_params must be > 0, got {per_element_output_params}"

        self.pseudoimage_x = pseudoimage_dims[0]
        self.pseudoimage_y = pseudoimage_dims[1]

        assert self.pseudoimage_x > 0, f"pseudoimage_x must be > 0, got {self.pseudoimage_x}"
        assert self.pseudoimage_y > 0, f"pseudoimage_y must be > 0, got {self.pseudoimage_y}"

        self.pseudoimage_stepdowns = np.ceil(
            np.max(np.log2(np.array([
                self.pseudoimage_x, self.pseudoimage_y
            ])))).astype(int) - latent_image_area_power
        print(f"pseudoimage_stepdowns: {self.pseudoimage_stepdowns}")

        self.sequence_length = sequence_length
        self.total_num_channels = per_element_num_channels * sequence_length

        self.latent_image_area = (2**latent_image_area_power)**2
        self.total_output_channels = np.ceil(
            per_element_output_params * sequence_length /
            self.latent_image_area).astype(int)
        print(f"output_channels: {self.total_output_channels}")
        self.reshaped_out_size = per_element_output_params * sequence_length

        self.conv_layers = nn.ModuleList()
        print(
            f"Appending first conv layer of {self.total_num_channels} channels to {self.total_output_channels} channels"
        )
        self.conv_layers.append(
            ConvWithNonLinearity(self.total_num_channels,
                                 self.total_output_channels, 3, 2, 1))
        for _ in range(self.pseudoimage_stepdowns - 2):
            print(
                f"Appending conv layer of {self.total_output_channels} channels to {self.total_output_channels} channels"
            )
            self.conv_layers.append(
                ConvWithNonLinearity(self.total_output_channels,
                                     self.total_output_channels, 3, 2, 1))
        self.conv_layers.append(
            nn.Conv2d(self.total_output_channels, self.total_output_channels,
                      3, 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor), f"x must be a torch.Tensor, got {type(x)}"
        assert x.shape[
            1] == self.total_num_channels, f"x must have shape (batch_size, {self.total_num_channels}, pseudoimage_x, pseudoimage_y), got {x.shape}"

        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = x.reshape(x.shape[0], -1)[:, :self.reshaped_out_size]
        return x
