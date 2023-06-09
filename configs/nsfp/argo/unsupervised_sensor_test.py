_base_ = "./unsupervised_sensor_train.py"

test_sequence_dir = "/efs/argoverse2/test/"
flow_save_folder = "/efs/argoverse2/test_nsfp_flow/"
save_output_folder = "/efs/argoverse2/test_nsfp_out/"

model = dict(args=dict(flow_save_folder=flow_save_folder))

test_loader = dict(args=dict(sequence_dir=test_sequence_dir))
