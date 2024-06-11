_base_ = "./test_lidar_camera.py"

save_output_folder = "/efs/argoverse2/test_gigachad_perf/"

model = dict(
    args=dict(lr=0.0000008, epochs=1, chamfer_distance_type="forward_only"),
)


test_dataset = dict(
    args=dict(log_subset=["af8471e6-6780-3df2-bc6a-1982a4b1b437"], subsequence_length=156)
)
