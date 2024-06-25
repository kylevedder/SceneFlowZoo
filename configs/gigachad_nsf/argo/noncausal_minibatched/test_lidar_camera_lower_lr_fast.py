_base_ = "./test_lidar_camera.py"

save_output_folder = "/efs/argoverse2/test_gigachad_nsf_lidar_camera_fast_flow_feather/"

model = dict(
    args=dict(
        lr=1e-5,
        chamfer_distance_type="forward_only",
        epochs=10000,
        scheduler=dict(
            name="ReduceLROnPlateauWithFloorRestart",
            args=dict(),
        ),
    ),
)
