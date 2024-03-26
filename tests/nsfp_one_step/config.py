is_trainable = False

test_dataset_root = "/tmp/argoverse2_tiny/val/"
save_output_folder = "/tmp/argoverse2_tiny/val_nsfp_out/"

epochs = 1
learning_rate = 2e-6
save_every = 500
validate_every = 10

SEQUENCE_LENGTH = 2

model = dict(
    name="NSFP",
    args=dict(
        sequence_length=SEQUENCE_LENGTH,
        iterations=1,
    ),
)

test_dataset = dict(
    name="BucketedSceneFlowDataset",
    args=dict(
        dataset_name="Argoverse2CausalSceneFlow",
        root_dir=test_dataset_root,
        with_ground=False,
        with_rgb=True,
        eval_type="bucketed_epe",
        eval_args=dict(),
    ),
)

test_dataloader = dict(args=dict(batch_size=1, num_workers=0, shuffle=False, pin_memory=True))
