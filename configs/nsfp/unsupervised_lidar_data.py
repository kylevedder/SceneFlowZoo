is_trainable = False
has_labels = False


test_sequence_dir = "/efs/argoverse_lidar/train/"
flow_save_folder = "/efs/argoverse_lidar/train_nsfp_flow/"

precision = 32


def get_max_sequence_length(sequence_dir):
    if "argoverse2" in sequence_dir:
        return 146
    else:
        return 296


max_test_sequence_length = get_max_sequence_length(test_sequence_dir)

epochs = 20
learning_rate = 2e-6
save_every = 500
validate_every = 500

SEQUENCE_LENGTH = 2

model = dict(name="NSFP",
             args=dict(VOXEL_SIZE=(0.2, 0.2, 4),
                       POINT_CLOUD_RANGE=(-51.2, -51.2, -3, 51.2, 51.2, 1),
                       SEQUENCE_LENGTH=SEQUENCE_LENGTH,
                       flow_save_folder=flow_save_folder))

test_loader = dict(name="ArgoverseSequenceLoader",
                   args=dict(sequence_dir=test_sequence_dir))

test_dataloader = dict(
    args=dict(batch_size=1, num_workers=1, shuffle=False, pin_memory=True))

test_dataset = dict(name="SubsequenceDataset",
                    args=dict(subsequence_length=SEQUENCE_LENGTH,
                              max_sequence_length=max_test_sequence_length,
                              origin_mode="FIRST_ENTRY"))
