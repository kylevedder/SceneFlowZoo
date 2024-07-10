_base_ = "./test.py"

save_output_folder = "/efs/argoverse2/test_gigachad_nsf_fast_feather/"

model = dict(
    args=dict(
        chamfer_distance_type="forward_only",
        epochs=2000,
    ),
)
