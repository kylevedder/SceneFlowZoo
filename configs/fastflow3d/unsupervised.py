train_sequence_dir = "/efs/argoverse_lidar/train/"

test_sequence_dir = "/efs/argoverse2/val/"
test_flow_dir = "/efs/argoverse2/val_sceneflow/"


def get_max_sequence_length(sequence_dir):
    if "argoverse2" in sequence_dir:
        return 146
    else:
        return 296


max_train_sequence_length = get_max_sequence_length(train_sequence_dir)
max_test_sequence_length = get_max_sequence_length(test_sequence_dir)

epochs = 20
learning_rate = 2e-6
save_every = 500
validate_every = 500

SEQUENCE_LENGTH = 2

model = dict(name="FastFlow3D",
             args=dict(VOXEL_SIZE=(0.2, 0.2, 4),
                       PSEUDO_IMAGE_DIMS=(512, 512),
                       POINT_CLOUD_RANGE=(-51.2, -51.2, -3, 51.2, 51.2, 1),
                       MAX_POINTS_PER_VOXEL=128,
                       FEATURE_CHANNELS=32,
                       SEQUENCE_LENGTH=SEQUENCE_LENGTH))

loader = dict(name="ArgoverseSequenceLoader",
              args=dict(sequence_dir=train_sequence_dir))

dataloader = dict(args=dict(batch_size=16, num_workers=8, shuffle=True, pin_memory=False))

dataset = dict(name="SubsequenceDataset",
               args=dict(subsequence_length=SEQUENCE_LENGTH,
                         max_sequence_length=max_train_sequence_length,
                         origin_mode="FIRST_ENTRY"))

loss_fn = dict(name="FastFlow3DUnsupervisedLoss", args=dict())

test_loader = dict(name="ArgoverseFlowSequenceLoader",
                   args=dict(raw_data_path=test_sequence_dir,
                             flow_data_path=test_flow_dir))

test_dataloader = dict(args=dict(batch_size=8, num_workers=8, shuffle=False, pin_memory=True))

test_dataset = dict(name="SubsequenceFlowDataset",
                    args=dict(subsequence_length=SEQUENCE_LENGTH,
                              max_sequence_length=max_test_sequence_length,
                              origin_mode="FIRST_ENTRY"))
