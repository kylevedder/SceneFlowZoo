_base_ = "./test_gaussian.py"

model = dict(
    args=dict(
        epochs=1,
        minibatch_size=4,
    ),
)

test_dataset_root = "/efs/argoverse2_small/val/"
save_output_folder = "/efs/argoverse2_small/val_throwaway_gigachad_occ_flow_feather/"

test_dataset = dict(
    args=dict(
        root_dir=test_dataset_root,
        log_subset=["02a00399-3857-444e-8db3-a8f58489c394"],
        subsequence_length=158,
    )
)
