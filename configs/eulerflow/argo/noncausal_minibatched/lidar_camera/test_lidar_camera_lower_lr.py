_base_ = "./test_lidar_camera.py"

save_output_folder = "/efs/argoverse2/test_eulerflow_lidar_camera_lower_lr_flow_feather/"

model = dict(
    args=dict(
        lr=0.0000008,
    ),
)
