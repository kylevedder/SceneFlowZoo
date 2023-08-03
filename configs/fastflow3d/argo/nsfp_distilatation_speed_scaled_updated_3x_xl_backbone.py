_base_ = "./nsfp_distilatation_speed_scaled_updated.py"

sensor_train_sequence_dir = "/efs/argoverse2/train/"
sensor_train_flow_dir = "/efs/argoverse2/train_nsfp_flow/"

lidar_train_sequence_dir = "/efs/argoverse_lidar_subsampled_all/train/"
lidar_train_flow_dir = "/efs/argoverse_lidar_subsampled_all/train_nsfp_flow/"

def get_max_sequence_length(sequence_dir):
    if "argoverse2" in sequence_dir:
        return 145
    else:
        return 2


max_sensor_train_sequence_length = get_max_sequence_length(
    sensor_train_sequence_dir)
max_lidar_train_sequence_length = get_max_sequence_length(
    lidar_train_sequence_dir)

SUBSEQUENCE_LENGTH = 2
POINT_CLOUD_RANGE = (-51.2, -51.2, -3, 51.2, 51.2, 3)
VOXEL_SIZE = (0.1, 0.1, POINT_CLOUD_RANGE[5] - POINT_CLOUD_RANGE[2])
PSEUDO_IMAGE_DIMS = (1024, 1024)

save_every = 500*8
validate_every = 500*8

model = dict(name="FastFlow3D",
             args=dict(VOXEL_SIZE=VOXEL_SIZE,
                       PSEUDO_IMAGE_DIMS=PSEUDO_IMAGE_DIMS,
                       FEATURE_CHANNELS=64,
                       xl_backbone=True))

dataloader = dict(
    args=dict(batch_size=2, num_workers=4, shuffle=True, pin_memory=True))

test_dataloader = dict(
    args=dict(batch_size=6, num_workers=12, shuffle=False, pin_memory=True))


loader = [
    dict(name="ArgoverseUnsupervisedFlowSequenceLoader",
         args=dict(raw_data_path=sensor_train_sequence_dir,
                   flow_data_path=sensor_train_flow_dir)),
    dict(name="ArgoverseUnsupervisedFlowSequenceLoader",
         args=dict(raw_data_path=lidar_train_sequence_dir,
                   flow_data_path=lidar_train_flow_dir))
]


dataset = [
    dict(name="SubsequenceUnsupervisedFlowDataset",
         args=dict(subsequence_length=SUBSEQUENCE_LENGTH,
                   max_sequence_length=max_sensor_train_sequence_length,
                   origin_mode="FIRST_ENTRY")),
    dict(name="SubsequenceUnsupervisedFlowDataset",
         args=dict(subsequence_length=SUBSEQUENCE_LENGTH,
                   max_sequence_length=max_lidar_train_sequence_length,
                   origin_mode="FIRST_ENTRY",
                   subset_fraction=1.0))
]
