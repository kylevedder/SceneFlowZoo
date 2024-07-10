_base_ = "../test.py"

save_output_folder = "/efs/argoverse2/test_gigachad_nsf_lidar_camera_flow_feather/"


model = dict(
    args=dict(
        chamfer_target_type="lidar_camera",
    ),
)
