from .nsfp_raw_mlp import NSFPRawMLP, ActivationFn


class GigaChadRawMLP(NSFPRawMLP):

    def __init__(
        self,
        input_dim: int = 5,
        output_dim: int = 3,
        latent_dim: int = 128,
        act_fn: ActivationFn = ActivationFn.RELU,
        num_layers: int = 8,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            act_fn=act_fn,
            num_layers=num_layers,
        )
