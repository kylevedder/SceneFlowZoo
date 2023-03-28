_base_ = "../../pseudoimage.py"

train_sequence_dir = "/efs/waymo_open_processed_flow/training/"

test_sequence_dir = "/efs/waymo_open_processed_flow/validation/"



max_train_sequence_length = 192
max_test_sequence_length = 192

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

loader = dict(name="WaymoSupervisedFlowSequenceLoader",
              args=dict(sequence_dir=train_sequence_dir))

dataloader = dict(
    args=dict(batch_size=16, num_workers=16, shuffle=True, pin_memory=False))

dataset = dict(name="SubsequenceSupervisedFlowDataset",
               args=dict(subsequence_length=SEQUENCE_LENGTH,
                         max_sequence_length=max_train_sequence_length,
                         origin_mode="FIRST_ENTRY"))

loss_fn = dict(name="FastFlow3DSupervisedLoss", args=dict())

test_loader = dict(name="WaymoSupervisedFlowSequenceLoader",
                   args=dict(sequence_dir=test_sequence_dir))

test_dataloader = dict(
    args=dict(batch_size=8, num_workers=8, shuffle=False, pin_memory=True))

test_dataset = dict(name="SubsequenceSupervisedFlowDataset",
                    args=dict(subsequence_length=SEQUENCE_LENGTH,
                              max_sequence_length=max_test_sequence_length,
                              origin_mode="FIRST_ENTRY"))
