_base_ = "../../pseudoimage.py"

is_trainable = False
has_labels = False

test_sequence_dir = "/efs/waymo_open_preprocessed/train/"
flow_save_folder = "/efs/waymo_open_preprocessed/train_nsfp_flow/"

precision = 32


def get_max_sequence_length(sequence_dir):
    return 195


max_test_sequence_length = get_max_sequence_length(test_sequence_dir)

epochs = 20
learning_rate = 2e-6
save_every = 500
validate_every = 500

SEQUENCE_LENGTH = 2

model = dict(name="NSFP",
             args=dict(VOXEL_SIZE={{_base_.VOXEL_SIZE}},
                       POINT_CLOUD_RANGE={{_base_.POINT_CLOUD_RANGE}},
                       SEQUENCE_LENGTH=SEQUENCE_LENGTH,
                       flow_save_folder=flow_save_folder))

test_loader = dict(name="WaymoRawSequenceLoader",
                   args=dict(sequence_dir=test_sequence_dir, verbose=True))

test_dataloader = dict(
    args=dict(batch_size=1, num_workers=1, shuffle=False, pin_memory=True))

test_dataset = dict(name="SubsequenceRawDataset",
                    args=dict(subsequence_length=SEQUENCE_LENGTH,
                              max_sequence_length=max_test_sequence_length,
                              origin_mode="FIRST_ENTRY"))
