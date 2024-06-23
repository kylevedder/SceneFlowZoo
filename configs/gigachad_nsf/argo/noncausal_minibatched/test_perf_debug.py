_base_ = "./test_lidar_camera.py"

save_output_folder = "/efs/argoverse2/test_gigachad_perf/"

model = dict(
    args=dict(
        lr=0.0000008,
        chamfer_distance_type="forward_only",
        epochs=1,
        checkpoint="/efs/argoverse2/test_gigachad_weights/be0615bc-1d82-334b-9c98-6adf40406955/opt_step_00004400_weights.pth",
        eval_only=True,
    ),
)


test_dataset = dict(
    args=dict(log_subset=["af8471e6-6780-3df2-bc6a-1982a4b1b437"], subsequence_length=156)
)
