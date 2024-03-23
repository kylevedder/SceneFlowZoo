_base_ = "./deflow_val.py"

has_labels = False

test_dataset_root = "/efs/argoverse2/test/"
save_output_folder = "/efs/argoverse2/test_deflow_flow/"

test_dataset = dict(args=dict(root_dir=test_dataset_root))