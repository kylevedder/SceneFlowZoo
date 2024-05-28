has_labels = False

SEQUENCE_LENGTH = 150

test_dataset_root = "/efs/argoverse2/test/"
save_output_folder = "/tmp/argoverse2/test_gigachad_nsf_flow_feather/"


model = dict(
    name="GigachadNSFOptimizationLoop",
    args=dict(
        save_flow_every=10,
        minibatch_size=5,
        speed_threshold=60.0 / 10.0,
        lr=0.00008,
        epochs=1000,
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
        eval_args=dict(log_subset=["fee0f78c-cf00-35c5-975b-72724f53fd64"], subsequence_length=156),
        subsequence_length=SEQUENCE_LENGTH,
        range_crop_type="ego",  # Ensures that the range is cropped to the ego vehicle, so points are not chopped off if the ego vehicle is driving large distances.
    ),
)


train_dataloader = dict(args=dict(batch_size=1, num_workers=0, shuffle=False, pin_memory=True))


test_dataset = train_dataset.copy()
test_dataloader = train_dataloader.copy()
