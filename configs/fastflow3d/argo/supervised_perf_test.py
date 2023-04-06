_base_ = "./supervised.py"

test_dataset = dict(
    name="VarLenSubsequenceSupervisedFlowSpecificSubsetDataset",
    args=dict(_delete_=True,
              subsequence_length={{_base_.SEQUENCE_LENGTH}},
              origin_mode="FIRST_ENTRY",
              subset_file="data_prep_scripts/argo/eval_sequence_id_index.txt"))

test_dataloader = dict(args=dict(batch_size=1))
