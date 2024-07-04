is_trainable = False
has_labels = False

SEQUENCE_LENGTH = 150

test_dataset_root = "/efs/argoverse2/train/"
save_output_folder = "/efs/argoverse2/train_ntp_cleaned_up_flow_feather/"

model = dict(
    name="NTPOptimizationLoop",
    args=dict(
        minibatch_size=4,
        lr=1e-5,
        epochs=1001,
        scheduler=dict(
            name="PassThroughScheduler",
            args=dict(schedule_name="StepLR", schedule_args=dict(step_size=500, gamma=0.5)),
        ),
    ),
)


train_dataset = dict(
    name="TorchFullFrameDataset",
    args=dict(
        dataset_name="Argoverse2NonCausalSceneFlow",
        root_dir=test_dataset_root,
        load_flow=False,
        with_ground=False,
        with_rgb=False,
        eval_type="bucketed_epe",
        eval_args=dict(),
        subsequence_length=SEQUENCE_LENGTH,
        range_crop_type="ego",  # Ensures that the range is cropped to the ego vehicle, so points are not chopped off if the ego vehicle is driving large distances.
    ),
)


train_dataloader = dict(args=dict(batch_size=1, num_workers=0, shuffle=False, pin_memory=True))


test_dataset = train_dataset.copy()
test_dataloader = train_dataloader.copy()
