_base_ = "./train.py"

test_dataset_root = "/efs/argoverse2_lidar_seq_len_160/train/"
save_output_folder = "/efs/argoverse2_lidar_seq_len_160/train_eulerflow_feather/"


test_dataset = dict(
    args=dict(
        root_dir=test_dataset_root,
    ),
)
