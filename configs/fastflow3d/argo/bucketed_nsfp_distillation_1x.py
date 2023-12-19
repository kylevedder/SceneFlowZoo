_base_ = "./bucketed_flow_test_av2_class_bucketed_epe.py"

train_sequence_dir = "/efs/argoverse2/train/"

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

train_dataset = dict(name="BucketedSceneFlowDataset",
                     args=dict(dataset_name="Argoverse2SceneFlow",
                               root_dir=train_sequence_dir,
                               with_ground=False,
                               use_gt_flow=False,
                               with_rgb=False,
                               eval_type="bucketed_epe",
                               eval_args=dict()))

train_dataloader = dict(
    args=dict(batch_size=16, num_workers=16, shuffle=True, pin_memory=False))

loss_fn = dict(name="FastFlow3DBucketedLoaderLoss", args=dict())

test_dataset = dict(args=dict(eval_args=dict(
    output_path="/tmp/frame_results/bucketed_epe/nsfp_distillation_1x/")))
