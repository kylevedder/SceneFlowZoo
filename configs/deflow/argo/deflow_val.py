_base_ = ["../../pseudoimage.py"]

has_labels = True
is_trainable = True

epochs = 50
learning_rate = 2e-6
save_every = 500
validate_every = 500

SEQUENCE_LENGTH = 2

model = dict(name="DeFlow",
             args=dict(VOXEL_SIZE={{_base_.VOXEL_SIZE}},
                       PSEUDO_IMAGE_DIMS={{_base_.PSEUDO_IMAGE_DIMS}},
                       POINT_CLOUD_RANGE={{_base_.POINT_CLOUD_RANGE}}))


loss_fn = dict(name="FastFlow3DBucketedLoaderLoss", args=dict())

test_dataset_root = "/efs/argoverse2/val/"
save_output_folder = "/efs/argoverse2/val_deflow_flow/"

test_dataset = dict(name="BucketedSceneFlowDataset",
                    args=dict(dataset_name="Argoverse2CausalSceneFlow", 
                              root_dir=test_dataset_root,
                              with_ground=False,
                              with_rgb=False,
                              eval_type="bucketed_epe",
                              eval_args=dict()))

test_dataloader = dict(
    args=dict(batch_size=1, num_workers=8, shuffle=False, pin_memory=True))

######## TRAIN DATASET ########

# train_sequence_dir = "/efs/argoverse2/train/"

# train_dataset = dict(name="BucketedSceneFlowDataset",
#                      args=dict(dataset_name="Argoverse2SceneFlow",
#                                root_dir=train_sequence_dir,
#                                with_ground=False,
#                                use_gt_flow=True,
#                                with_rgb=False,
#                                eval_type="bucketed_epe",
#                                eval_args=dict()))

# train_dataloader = dict(
#     args=dict(batch_size=16, num_workers=16, shuffle=True, pin_memory=False))
