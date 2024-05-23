has_labels = False

SEQUENCE_LENGTH = 30

test_dataset_root = "/efs/argoverse2/val/"
save_output_folder = "/efs/argoverse2/val_point_tracker_3d_feather/"

camera_names = [
    "ring_front_center",
    "ring_rear_right",
    "ring_rear_left",
]

model = dict(
    name="PointTracker3D",
    args=dict(camera_names=camera_names),
)

train_dataset = dict(
    name="RawFullFrameDataset",
    args=dict(
        dataset_name="Argoverse2NonCausalSceneFlow",
        root_dir=test_dataset_root,
        with_ground=False,
        with_rgb=True,
        eval_type="bucketed_epe",
        eval_args=dict(),
        subsequence_length=SEQUENCE_LENGTH,
        camera_names=camera_names,
        split=dict(split_idx=1, num_splits=150 * 5),
    ),
)


train_dataloader = dict(args=dict(batch_size=1, num_workers=0, shuffle=False, pin_memory=True))


test_dataset = train_dataset.copy()
test_dataloader = train_dataloader.copy()
