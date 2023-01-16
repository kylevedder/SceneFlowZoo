import torch
import torch.nn as nn
# Import functional
import torch.nn.functional as F


class NeuralSceneFlowPriorOptimizable(nn.Module):

    def __init__(self,
                 params: torch.Tensor,
                 num_hidden_units: int = 64,
                 num_hidden_layers: int = 4,
                 nonlinearity=nn.Sigmoid()):
        super().__init__()

        self.num_hidden_units = num_hidden_units
        self.num_hidden_layers = num_hidden_layers

        # Decode the network into linears and biases.
        self.module_list = nn.ModuleList()
        linear_nn, idx_offset = self._decode_linear_from_params(
            params, 3, self.num_hidden_units, 0)
        self.module_list.append(linear_nn)
        self.module_list.append(nonlinearity)
        for _ in range(self.num_hidden_layers - 1):
            linear_nn, idx_offset = self._decode_linear_from_params(
                params, self.num_hidden_units, self.num_hidden_units,
                idx_offset)
            self.module_list.append(linear_nn)
            self.module_list.append(nonlinearity)
        linear_nn, idx_offset = self._decode_linear_from_params(
            params, self.num_hidden_units, 3, idx_offset)
        self.module_list.append(linear_nn)
        self.module_list.append(nonlinearity)

    def _decode_linear_from_params(self, params: torch.Tensor, in_units: int,
                                   out_units: int, idx_offset: int):
        # Params are K, where K is the number of parameters in the network
        assert isinstance(
            params,
            torch.Tensor), f"params must be a torch.Tensor, got {type(params)}"
        linear_nn = nn.Linear(out_units, in_units, bias=True)
        with torch.no_grad():
            # The first idx_offset parameters are for the first linear layer
            # The next in_units * out
            # The next out_units parameters are for the bias
            linear_size = in_units * out_units
            linear = params[idx_offset:idx_offset + linear_size]
            bias = params[idx_offset + linear_size:idx_offset + linear_size +
                          out_units]
            linear = linear.reshape(out_units, in_units)

            # assign the results to the linear module.
            linear_nn.weight = nn.Parameter(linear)
            linear_nn.bias = nn.Parameter(bias)
        return linear_nn, idx_offset + linear_size + out_units

    def _decode_params_from_linear(self):
        param_list = []
        for module in self.module_list:
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    param_list.append(module.weight.reshape(-1).detach())
                    param_list.append(module.bias.detach())
        return torch.cat(param_list, dim=0)

    def decode_params(self):
        return self._decode_params_from_linear()

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # Points are N x 3
        assert isinstance(
            points,
            torch.Tensor), f"points must be a torch.Tensor, got {type(points)}"
        assert points.shape[
            1] == 3, f"points must be N x 3, got {points.shape}"
        points = torch.unsqueeze(points, dim=0)
        for module in self.module_list:
            points = module(points)
        points = torch.squeeze(points, dim=0)
        return points
