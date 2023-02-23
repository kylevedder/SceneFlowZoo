_base_ = "./nsfp_distilatation.py"

sensor_train_sequence_dir = "/efs/argoverse2/train/"
sensor_train_flow_dir = "/efs/argoverse2/train_nsfp_flow/"

lidar_train_sequence_dir = "/efs/argoverse_lidar/train/"
lidar_train_flow_dir = "/efs/argoverse_lidar_subset/train_nsfp_flow/"


def get_max_sequence_length(sequence_dir):
    if "argoverse2" in sequence_dir:
        return 145
    else:
        return 294


max_sensor_train_sequence_length = get_max_sequence_length(
    sensor_train_sequence_dir)
max_lidar_train_sequence_length = get_max_sequence_length(
    lidar_train_sequence_dir)

epochs = 50
SUBSEQUENCE_LENGTH = 2

loader = [
    dict(name="ArgoverseUnsupervisedFlowSequenceLoader",
         args=dict(raw_data_path=sensor_train_sequence_dir,
                   flow_data_path=sensor_train_flow_dir)),
    dict(name="ArgoverseUnsupervisedFlowSequenceLoader",
         args=dict(raw_data_path=lidar_train_sequence_dir,
                   flow_data_path=lidar_train_flow_dir))
]

# The sensor dataset is 3600 batches from 750 * 145 sequences
# 3600 / (750 * 145) = 0.0331034483
# 3600  = 0.0331034483 * (750 * 145)

# We want to match this with the lidar dataset.
# 3600 = X * (2500 * 294) * 0.0331034483
# 3600 / (2500 * 294 * 0.0331034483) = 0.1479


dataset = [
    dict(name="SubsequenceUnsupervisedFlowDataset",
         args=dict(subsequence_length=SUBSEQUENCE_LENGTH,
                   max_sequence_length=max_sensor_train_sequence_length,
                   origin_mode="FIRST_ENTRY")),
    dict(name="SubsequenceUnsupervisedFlowDataset",
         args=dict(subsequence_length=SUBSEQUENCE_LENGTH,
                   max_sequence_length=max_lidar_train_sequence_length,
                   origin_mode="FIRST_ENTRY",
                   subset_fraction=0.1479))
]
