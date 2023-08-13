_base_ = "./supervised.py"

save_output_folder = "/efs/argoverse2/val_supervised_out/"

test_dataloader = dict(
    args=dict(batch_size=16, num_workers=16, shuffle=False, pin_memory=True)) 

test_dataset = dict(name="VarLenSubsequenceSupervisedFlowDataset",
                    args=dict(_delete_=True,
                              subsequence_length={{_base_.SEQUENCE_LENGTH}},
                              origin_mode="FIRST_ENTRY",
                              max_pc_points=120000))