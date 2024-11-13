_base_ = ["../../pseudoimage.py"]

epochs = 50
learning_rate = 2e-6
save_every = 500
validate_every = 500

SEQUENCE_LENGTH = 2

save_output_folder = "/efs/waymo_open_processed_flow/val_fastflow3d_feather/"

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

######## TEST DATASET ########

test_dataset_root = "/efs/waymo_open_processed_flow/validation/"

test_dataset = dict(
    name="TorchFullFrameDataset",
    args=dict(
        dataset_name="WaymoOpenCausalSceneFlow",
        root_dir=test_dataset_root,
        flow_folder=None,
        with_rgb=False,
        eval_type="bucketed_epe",
        max_pc_points=180000,
        allow_pc_slicing=True,
        eval_args=dict(output_path="eval_results/bucketed_epe/waymo/fastflow3d_supervised/"),
    ),
)

test_dataloader = dict(args=dict(batch_size=8, num_workers=8, shuffle=False, pin_memory=True))

######## TRAIN DATASET ########

train_sequence_dir = "/efs/waymo_open_processed_flow/training/"

train_dataset = dict(
    name="TorchFullFrameDataset",
    args=dict(
        dataset_name="WaymoOpenCausalSceneFlow",
        root_dir=train_sequence_dir,
        flow_folder=None,
        use_gt_flow=False,
        with_rgb=False,
        eval_type="bucketed_epe",
        max_pc_points=150000,
        allow_pc_slicing=True,
        eval_args=dict(),
    ),
)

train_dataloader = dict(args=dict(batch_size=16, num_workers=16, shuffle=True, pin_memory=False))
