_base_ = ["../../pseudoimage.py"]

epochs = 50
learning_rate = 2e-6
save_every = 500
validate_every = 500

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

######## TEST DATASET ########

test_dataset_root = "/efs/argoverse2/val/"

test_dataset = dict(
    name="Flow4DSceneFlowDataset",
    args=dict(
        dataset_name="Argoverse2CausalSceneFlow",
        root_dir=test_dataset_root,
        with_ground=False,
        with_rgb=False,
        eval_type="bucketed_epe",
        expected_camera_shape=(194, 256, 3),
        # point_cloud_range=None,
        eval_args=dict(output_path="eval_results/bucketed_epe/nsfp_distillation_1x/"),
    ),
)

test_dataloader = dict(args=dict(batch_size=8, num_workers=8, shuffle=False, pin_memory=True))

######## TRAIN DATASET ########

train_sequence_dir = "/efs/argoverse2/train/"

train_dataset = dict(
    name="BucketedSceneFlowDataset",
    args=dict(
        dataset_name="Argoverse2CausalSceneFlow",
        root_dir=train_sequence_dir,
        with_ground=False,
        use_gt_flow=False,
        with_rgb=False,
        eval_type="bucketed_epe",
        expected_camera_shape=(194, 256, 3),
        # point_cloud_range=None,
        eval_args=dict(),
    ),
)

train_dataloader = dict(args=dict(batch_size=16, num_workers=16, shuffle=True, pin_memory=False))
