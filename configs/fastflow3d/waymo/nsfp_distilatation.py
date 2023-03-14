train_sequence_dir = "/efs/waymo_open_preprocessed/train/"
train_flow_dir = "/efs/waymo_open_preprocessed/train_nsfp_flow/"

test_sequence_dir = "/efs/waymo_open_preprocessed/val/"


def get_max_sequence_length(sequence_dir):
    return 145
    


max_train_sequence_length = get_max_sequence_length(train_sequence_dir)
max_test_sequence_length = get_max_sequence_length(test_sequence_dir)

epochs = 50
learning_rate = 2e-6
save_every = 500
validate_every = 5

SEQUENCE_LENGTH = 2

model = dict(name="FastFlow3D",
             args=dict(VOXEL_SIZE=(0.2, 0.2, 4),
                       PSEUDO_IMAGE_DIMS=(512, 512),
                       POINT_CLOUD_RANGE=(-51.2, -51.2, -3, 51.2, 51.2, 1),
                       FEATURE_CHANNELS=32,
                       SEQUENCE_LENGTH=SEQUENCE_LENGTH))

loader = dict(name="WaymoUnsupervisedFlowSequenceLoader",
              args=dict(raw_data_path=train_sequence_dir,
                        flow_data_path=train_flow_dir))

dataloader = dict(
    args=dict(batch_size=16, num_workers=16, shuffle=False, pin_memory=False))

dataset = dict(name="SubsequenceUnsupervisedFlowDataset",
               args=dict(subsequence_length=SEQUENCE_LENGTH,
                         max_sequence_length=max_train_sequence_length,
                         origin_mode="FIRST_ENTRY"))

loss_fn = dict(name="FastFlow3DDistillationLoss", args=dict())

test_loader = dict(name="WaymoSupervisedFlowSequenceLoader",
                   args=dict(sequence_dir=test_sequence_dir))

test_dataloader = dict(
    args=dict(batch_size=8, num_workers=8, shuffle=False, pin_memory=True))

test_dataset = dict(name="SubsequenceSupervisedFlowDataset",
                    args=dict(subsequence_length=SEQUENCE_LENGTH,
                              max_sequence_length=max_test_sequence_length,
                              origin_mode="FIRST_ENTRY"))
