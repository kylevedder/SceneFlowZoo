import torch
import torch.nn as nn
import enum
from dataloaders import TorchFullFrameInputSequence, TorchFullFrameOutputSequence
from models.components.optimization.cost_functions import TruncatedChamferLossProblem


class ActivationFn(enum.Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    SINC = "sinc"  # https://openreview.net/forum?id=0Lqyut1y7M
    GAUSSIAN = "gaussian"  # https://arxiv.org/abs/2204.05735


class SinC(nn.Module):
    def __init__(self):
        super(SinC, self).__init__()

    def forward(self, x):
        return torch.sinc(x)


class Gaussian(nn.Module):
    def __init__(
        self,
        sigma: float = 0.1,  # GARF default value
    ):
        super(Gaussian, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        # From https://github.com/sfchng/Gaussian-Activated-Radiance-Fields/blob/74d72387bb2526755a8d6c07f6f900ec6a1be594/model/nerf_gaussian.py#L457-L464
        return (-0.5 * (x) ** 2 / self.sigma**2).exp()


class NSFPRawMLP(nn.Module):

    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 3,
        latent_dim: int = 128,
        act_fn: ActivationFn = ActivationFn.RELU,
        num_layers: int = 8,
        with_compile: bool = True,
    ):
        super().__init__()
        self.layer_size = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.act_fn = act_fn
        self.nn_layers = torch.nn.Sequential(*self._make_model())
        if with_compile:
            self.nn_layers = torch.compile(self.nn_layers, dynamic=True)

    def _get_activation_fn(self) -> nn.Module:
        match self.act_fn:
            case ActivationFn.RELU:
                return torch.nn.ReLU()
            case ActivationFn.SIGMOID:
                return torch.nn.Sigmoid()
            case ActivationFn.SINC:
                return SinC()
            case ActivationFn.GAUSSIAN:
                return Gaussian()
            case _:
                raise ValueError(f"Unsupported activation function: {self.act_fn}")

    def _make_model(self) -> torch.nn.ModuleList:
        nn_layers = torch.nn.ModuleList([])
        if self.layer_size <= 1:
            nn_layers.append(torch.nn.Sequential(torch.nn.Linear(self.input_dim, self.output_dim)))
            return nn_layers

        nn_layers.append(torch.nn.Sequential(torch.nn.Linear(self.input_dim, self.latent_dim)))
        nn_layers.append(self._get_activation_fn())
        for _ in range(self.layer_size - 1):
            nn_layers.append(torch.nn.Sequential(torch.nn.Linear(self.latent_dim, self.latent_dim)))
            nn_layers.append(self._get_activation_fn())
        nn_layers.append(torch.nn.Linear(self.latent_dim, self.output_dim))

        return nn_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn_layers(x)
