_base_ = "./val.py"

test_dataset_root = "/efs/argoverse2_small/val/"
save_output_folder = "/efs/argoverse2_small/val_gigachad_nsf_flow_feather/"

SEQUENCE_LENGTH = 16

test_dataset = dict(
    args=dict(
        root_dir=test_dataset_root,
        subsequence_length=SEQUENCE_LENGTH,
        split=dict(split_idx=25 // 2, num_splits=157),
    )
)
# test_dataset = dict(args=dict(root_dir=test_dataset_root, subsequence_length=SEQUENCE_LENGTH))


model = dict(args=dict(save_flow_every=10))
