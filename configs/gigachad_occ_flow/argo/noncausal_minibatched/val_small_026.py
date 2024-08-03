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
        log_subset=["02678d04-cc9f-3148-9f95-1ba66347dff9"],
        subsequence_length=157,
    )
)
