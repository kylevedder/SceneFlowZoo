_base_ = ["../../pseudoimage.py"]

epochs = 50
learning_rate = 2e-6
save_every = 500
validate_every = 500

SEQUENCE_LENGTH = 2

model = dict(name="FastFlow3D",
             args=dict(VOXEL_SIZE={{_base_.VOXEL_SIZE}},
                       PSEUDO_IMAGE_DIMS={{_base_.PSEUDO_IMAGE_DIMS}},
                       POINT_CLOUD_RANGE={{_base_.POINT_CLOUD_RANGE}},
                       FEATURE_CHANNELS=32,
                       SEQUENCE_LENGTH=SEQUENCE_LENGTH))

loss_fn = dict(name="FastFlow3DBucketedLoaderLoss", args=dict())

###############################################################################

test_dataset_root = "/efs/argoverse2/val/"

test_dataset = dict(
    name="BucketedSceneFlowDataset",
    args=dict(dataset_name="Argoverse2SceneFlow",
              root_dir=test_dataset_root,
              with_ground=False,
              with_rgb=False,
              eval_type="bucketed_epe",
              eval_args=dict()))

test_dataloader = dict(
    args=dict(batch_size=8, num_workers=8, shuffle=False, pin_memory=True))