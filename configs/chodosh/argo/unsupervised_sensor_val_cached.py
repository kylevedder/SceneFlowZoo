_base_ = "./unsupervised_sensor_val_dumper.py"

has_labels = True
test_sequence_dir = "/efs/argoverse2/val/"
test_flow_dir = "/efs/argoverse2/val_sceneflow/"

flow_save_folder = "/efs/argoverse2/val_chodosh_flow/"

test_loader = dict(name="ArgoverseSupervisedFlowSequenceLoader",
                   args=dict(_delete_=True,
                             raw_data_path=test_sequence_dir,
                             flow_data_path=test_flow_dir))
test_dataset = dict(name="SubsequenceSupervisedFlowDataset")
model = dict(name="NSFPCached", args=dict(_delete_=True, VOXEL_SIZE={{_base_.VOXEL_SIZE}},
                       POINT_CLOUD_RANGE={{_base_.POINT_CLOUD_RANGE}},
                       SEQUENCE_LENGTH={{_base_.SEQUENCE_LENGTH}},
                       flow_save_folder=flow_save_folder))