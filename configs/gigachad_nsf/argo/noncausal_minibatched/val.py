_base_ = "./test.py"

test_dataset_root = "/efs/argoverse2/val/"
save_output_folder = "/efs/argoverse2/val_gigachad_nsf_feather/"


test_dataset = dict(
    args=dict(
        root_dir=test_dataset_root,
    ),
)
