_base_ = "./test.py"

save_output_folder = "/efs/argoverse2/test_gigachad_occ_flow_gaussian_feather/"

model = dict(
    name="GigachadOccFlowGaussianOptimizationLoop",
    args=dict(
        minibatch_size=4,
    ),
)
