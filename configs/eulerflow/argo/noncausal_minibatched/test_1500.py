_base_ = "./test.py"

save_output_folder = "/efs/argoverse2/test_eulerflow_kd_forward_backward_1500_feather/"

model = dict(
    args=dict(
        epochs=1500,
    ),
)
