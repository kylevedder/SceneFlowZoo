_base_ = './unsupervised_lidar_data_subset.py'

test_sequence_dir = "/efs/argoverse_lidar_subset_compliment/train/"
flow_save_folder = "/efs/argoverse_lidar_subset_compliment/train_nsfp_flow/"

model = dict(args=dict(flow_save_folder=flow_save_folder))

test_loader = dict(args=dict(sequence_dir=test_sequence_dir))
