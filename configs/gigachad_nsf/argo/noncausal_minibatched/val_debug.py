has_labels = False

SEQUENCE_LENGTH = 150

test_dataset_root = "/efs/argoverse2_small/val/"
save_output_folder = "/efs/argoverse2_small/val_gigachad_nsf_flow_feather/"


model = dict(
    name="GigachadNSFOptimizationLoop",
    args=dict(
        save_flow_every=1,
        minibatch_size=10,
        speed_threshold=60.0 / 10.0,
        lr=0.008,
    ),
)

train_dataset = dict(
    name="BucketedSceneFlowDataset",
    args=dict(
        dataset_name="Argoverse2NonCausalSceneFlow",
        root_dir=test_dataset_root,
        with_ground=False,
        with_rgb=False,
        eval_type="bucketed_epe",
        eval_args=dict(),
        subsequence_length=SEQUENCE_LENGTH,
        split=dict(split_idx=0, num_splits=2),
    ),
)


train_dataloader = dict(args=dict(batch_size=1, num_workers=0, shuffle=False, pin_memory=True))


test_dataset = train_dataset.copy()
test_dataloader = train_dataloader.copy()
