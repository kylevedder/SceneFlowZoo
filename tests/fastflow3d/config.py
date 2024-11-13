_base_ = ["../../configs/pseudoimage.py"]

epochs = 1
learning_rate = 2e-6
save_every = 500
validate_every = 10

SEQUENCE_LENGTH = 2

model = dict(
    name="FastFlow3D",
    args=dict(
        VOXEL_SIZE={{_base_.VOXEL_SIZE}},
        PSEUDO_IMAGE_DIMS={{_base_.PSEUDO_IMAGE_DIMS}},
        POINT_CLOUD_RANGE={{_base_.POINT_CLOUD_RANGE}},
        FEATURE_CHANNELS=32,
        SEQUENCE_LENGTH=SEQUENCE_LENGTH,
    ),
)

test_dataset_root = "/tmp/argoverse2_tiny/val/"
save_output_folder = "/tmp/argoverse2_tiny/val_nsfp_out/"

epochs = 1
learning_rate = 2e-6
save_every = 500
validate_every = 10

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
