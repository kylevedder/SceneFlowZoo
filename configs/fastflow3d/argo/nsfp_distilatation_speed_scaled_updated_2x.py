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
