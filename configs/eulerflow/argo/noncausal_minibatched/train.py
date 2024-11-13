_base_ = "./val_depth18.py"

test_dataset_root = "/efs/argoverse2/train/"
save_output_folder = "/efs/argoverse2/train_eulerflow_feather/"


test_dataset = dict(
    args=dict(
        root_dir=test_dataset_root,
    ),
)
