is_trainable = False

test_dataset_root = "/tmp/argoverse2_small/val/"
save_output_folder = "/tmp/argoverse2_small/val_liu_2024_out/"

epochs = 1
learning_rate = 2e-6
save_every = 500
validate_every = 10

SEQUENCE_LENGTH = 3

model = dict(
    name="Liu2024Model",
    args=dict(
        iterations=10,
    ),
)

test_dataset = dict(
    name="BucketedSceneFlowDataset",
    args=dict(
        dataset_name="Argoverse2CausalSceneFlow",
        root_dir=test_dataset_root,
        with_ground=False,
        with_rgb=False,
        eval_type="bucketed_epe",
        eval_args=dict(),
        subsequence_length=SEQUENCE_LENGTH,
        set_length=2,
    ),
)

test_dataloader = dict(args=dict(batch_size=1, num_workers=0, shuffle=False, pin_memory=True))
