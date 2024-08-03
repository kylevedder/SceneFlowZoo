_base_ = "./test.py"

save_output_folder = "/efs/argoverse2/test_gigachad_occ_flow_sinc_depth_14_feather/"

model = dict(
    name="GigachadOccFlowSincDepth14OptimizationLoop",
    args=dict(
        minibatch_size=3,
    ),
)
