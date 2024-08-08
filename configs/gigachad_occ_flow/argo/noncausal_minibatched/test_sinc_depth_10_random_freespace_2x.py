_base_ = "./test.py"

save_output_folder = (
    "/efs/argoverse2/test_gigachad_occ_flow_sinc_depth_10_random_freespace_2x_feather/"
)

model = dict(
    name="GigachadOccFlowSincDepth10RandomSampleFreespace2xOptimizationLoop",
    args=dict(
        minibatch_size=3,
        epochs=3000,
    ),
)
