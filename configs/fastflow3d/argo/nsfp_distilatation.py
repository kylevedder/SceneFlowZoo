_base_ = "../../pseudoimage.py"

train_sequence_dir = "/efs/argoverse2/train/"
train_flow_dir = "/efs/argoverse2/train_nsfp_flow/"

test_sequence_dir = "/efs/argoverse2/val/"
test_flow_dir = "/efs/argoverse2/val_sceneflow/"


def get_max_sequence_length(sequence_dir):
    if "argoverse2" in sequence_dir:
        return 145
    else:
        return 296


max_train_sequence_length = get_max_sequence_length(train_sequence_dir)
max_test_sequence_length = get_max_sequence_length(test_sequence_dir)

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

loader = dict(name="ArgoverseUnsupervisedFlowSequenceLoader",
              args=dict(raw_data_path=train_sequence_dir,
                        flow_data_path=train_flow_dir))

dataloader = dict(
    args=dict(batch_size=16, num_workers=16, shuffle=True, pin_memory=False))

dataset = dict(name="SubsequenceUnsupervisedFlowDataset",
               args=dict(subsequence_length=SEQUENCE_LENGTH,
                         max_sequence_length=max_train_sequence_length,
                         origin_mode="FIRST_ENTRY"))

loss_fn = dict(name="FastFlow3DDistillationLoss", args=dict())

test_loader = dict(name="ArgoverseSupervisedFlowSequenceLoader",
                   args=dict(raw_data_path=test_sequence_dir,
                             flow_data_path=test_flow_dir))

test_dataloader = dict(
    args=dict(batch_size=8, num_workers=8, shuffle=False, pin_memory=True))

test_dataset = dict(name="SubsequenceSupervisedFlowDataset",
                    args=dict(subsequence_length=SEQUENCE_LENGTH,
                              max_sequence_length=max_test_sequence_length,
                              origin_mode="FIRST_ENTRY"))
