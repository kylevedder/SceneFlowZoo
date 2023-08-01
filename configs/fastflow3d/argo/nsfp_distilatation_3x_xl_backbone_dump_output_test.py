_base_ = "./nsfp_distilatation_3x_xl_backbone.py"

has_labels = False

test_sequence_dir = "/efs/argoverse2/test/"
test_flow_dir = "/efs/argoverse2/test_sceneflow/"

save_output_folder = "/efs/argoverse2/test_nsfp_distilation_3x_xl_backbone_out/"
SEQUENCE_LENGTH = 2

test_loader = dict(_delete_=True,
                   name="ArgoverseRawSequenceLoader",
                   args=dict(sequence_dir=test_sequence_dir, verbose=True))

test_dataset = dict(name="VarLenSubsequenceRawDataset",
                    args=dict(_delete_=True,
                              subsequence_length=SEQUENCE_LENGTH,
                              origin_mode="FIRST_ENTRY",
                              max_pc_points=120000))