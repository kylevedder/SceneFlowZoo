_base_ = "./train.py"

test_dataset_root = "/efs/argoverse2/test/"
save_output_folder = "/efs/argoverse2/test_fast_nsf_plus_plus_flow_feather/"

test_dataset = dict(args=dict(root_dir=test_dataset_root, load_flow=False))
