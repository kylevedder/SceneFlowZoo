_base_ = "./test.py"

save_output_folder = "/efs/argoverse2/test_eulerflow_kd_forward_only_feather/"

model = dict(
    args=dict(
        pc_loss_type="truncated_kd_tree_forward",
        epochs=2000,
    ),
)
