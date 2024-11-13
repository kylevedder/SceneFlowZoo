_base_ = "./val.py"

test_dataset_root = "/efs/argoverse2_seq_len_5/val/"
save_output_folder = "/efs/argoverse2_seq_len_5/val_eulerflow_feather/"


test_dataset = dict(
    args=dict(
        root_dir=test_dataset_root,
    ),
)
