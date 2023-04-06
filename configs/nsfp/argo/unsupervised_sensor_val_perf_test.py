_base_ = "./unsupervised_sensor_val.py"

flow_save_folder = "/tmp/nsfp_perf_test_results/"

has_labels = True
test_sequence_dir = "/efs/argoverse2/val/"
test_flow_dir = "/efs/argoverse2/val_sceneflow/"

test_loader = dict(name="ArgoverseSupervisedFlowSequenceLoader",
                   args=dict(_delete_=True,
                             raw_data_path=test_sequence_dir,
                             flow_data_path=test_flow_dir))

test_dataset = dict(
    name="VarLenSubsequenceSupervisedFlowSpecificSubsetDataset",
    args=dict(_delete_=True,
              subsequence_length={{_base_.SEQUENCE_LENGTH}},
              origin_mode="FIRST_ENTRY",
              subset_file="data_prep_scripts/argo/eval_sequence_id_index.txt"))

model = dict(args=dict(flow_save_folder=flow_save_folder))
