is_trainable = False
has_labels = False

test_dataset_root = "/tmp/argoverse2_small/val/"
save_output_folder = "/tmp/argoverse2_small/val_ntp_out/"

epochs = 1
learning_rate = 0.003
save_every = 500
validate_every = 10

SEQUENCE_LENGTH = 4

model = dict(
    name="NTPOptimizationLoop",
    args=dict(
        minibatch_size=4,
        lr=0.003,
        epochs=1,
        scheduler=dict(
            name="PassThroughScheduler",
            args=dict(schedule_name="StepLR", schedule_args=dict(step_size=500, gamma=0.5)),
        ),
    ),
)

test_dataset = dict(
    name="TorchFullFrameDataset",
    args=dict(
        dataset_name="Argoverse2NonCausalSceneFlow",
        root_dir=test_dataset_root,
        with_ground=False,
        with_rgb=False,
        eval_type="bucketed_epe",
        eval_args=dict(),
        subsequence_length=SEQUENCE_LENGTH,
        set_length=1,
    ),
)

test_dataloader = dict(args=dict(batch_size=1, num_workers=0, shuffle=False, pin_memory=True))
