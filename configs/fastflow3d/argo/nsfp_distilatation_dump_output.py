_base_ = "./nsfp_distilatation.py"

has_labels = False

test_sequence_dir = "/efs/argoverse2/val/"
test_flow_dir = "/efs/argoverse2/val_sceneflow/"

save_output_folder = "/efs/argoverse2/val_distilation_out/"
SEQUENCE_LENGTH = 2

test_loader = dict(_delete_=True,
                   name="ArgoverseRawSequenceLoader",
                   args=dict(sequence_dir=test_sequence_dir, verbose=True))

test_dataset = dict(name="VarLenSubsequenceRawDataset",
                    args=dict(_delete_=True,
                              subsequence_length=SEQUENCE_LENGTH,
                              origin_mode="FIRST_ENTRY"))

test_dataloader = dict(
    args=dict(batch_size=16, num_workers=16, shuffle=False, pin_memory=True))