import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralSceneFlowPrior(nn.Module):

    def __init__(self, num_hidden_units=64, num_hidden_layers=4):
        super().__init__()
        assert num_hidden_units > 0, f"num_hidden_units must be > 0, got {num_hidden_units}"
        assert num_hidden_layers > 0, f"num_hidden_layers must be > 0, got {num_hidden_layers}"
        param_count = self._num_params_for_linear(3, num_hidden_units)
        for _ in range(num_hidden_layers - 1):
            param_count += self._num_params_for_linear(num_hidden_units,
                                                       num_hidden_units)
        param_count += self._num_params_for_linear(num_hidden_units, 3)
        self.param_count = param_count
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_units = num_hidden_units

    def _num_params_for_linear(self, input_dim, output_dim):
        # Rectangular matrix plus bias
        return input_dim * output_dim + output_dim

    def decode_linear_from_params(self, params: torch.Tensor, in_units: int,
                                  out_units: int, idx_offset: int):
        # Params are K, where K is the number of parameters in the network
        assert isinstance(
            params,
            torch.Tensor), f"params must be a torch.Tensor, got {type(params)}"
        assert params.shape == (
            self.param_count,
        ), f"params must have shape ({self.param_count},), got {params.shape}"

        # The first idx_offset parameters are for the first linear layer
        # The next in_units * out
        # The next out_units parameters are for the bias
        linear_size = in_units * out_units
        linear = params[idx_offset:idx_offset + linear_size]
        bias = params[idx_offset + linear_size:idx_offset + linear_size +
                      out_units]
        linear = linear.reshape(in_units, out_units)
        return linear, bias, idx_offset + linear_size + out_units

    def forward(self, points: torch.Tensor,
                params: torch.Tensor) -> torch.Tensor:

        # Points are N x 3
        # Params are K, where K is the number of parameters in the network
        assert isinstance(
            points,
            torch.Tensor), f"x must be a torch.Tensor, got {type(points)}"
        assert points.shape[
            1] == 3, f"x must have shape (points, 3), got {points.shape}"

        assert isinstance(
            params,
            torch.Tensor), f"params must be a torch.Tensor, got {type(params)}"
        assert params.shape == (
            self.param_count,
        ), f"params must have shape ({self.param_count},), got {params.shape}"

        # Decode the network into linears and biases.
        linears = []
        biases = []
        linear, bias, idx_offset = self.decode_linear_from_params(
            params, 3, self.num_hidden_units, 0)
        linears.append(linear)
        biases.append(bias)
        for _ in range(self.num_hidden_layers - 1):
            linear, bias, idx_offset = self.decode_linear_from_params(
                params, self.num_hidden_units, self.num_hidden_units,
                idx_offset)
            linears.append(linear)
            biases.append(bias)
        linear, bias, idx_offset = self.decode_linear_from_params(
            params, self.num_hidden_units, 3, idx_offset)
        linears.append(linear)
        biases.append(bias)
        assert idx_offset == self.param_count, f"idx_offset must be {self.param_count}, got {idx_offset}"

        # Evaluate the MLP on the points
        for idx, (linear, bias) in enumerate(zip(linears, biases)):
            points = torch.matmul(points, linear) + bias
            if idx < len(linears) - 1:
                points = F.gelu(points)
        return points