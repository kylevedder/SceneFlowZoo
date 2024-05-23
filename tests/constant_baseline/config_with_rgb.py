is_trainable = False

test_dataset_root = "/tmp/argoverse2_tiny/val/"
save_output_folder = "/tmp/argoverse2_tiny/val_constant_baseline_out/"

epochs = 1
learning_rate = 2e-6

SEQUENCE_LENGTH = 2

model = dict(
    name="ConstantVectorBaseline",
    args=dict(
        default_vector=[1.0, 2.0, 3.0],
    ),
)

test_dataset = dict(
    name="TorchFullFrameDataset",
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
