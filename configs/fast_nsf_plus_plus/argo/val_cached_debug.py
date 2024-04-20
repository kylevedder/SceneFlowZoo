_base_ = "./val_cached.py"

test_dataset_root = "/efs/argoverse2_small/val/"
save_output_folder = "/efs/argoverse2_small/val_fast_nsf_plus_plus_flow_feather/"

test_dataset = dict(args=dict(root_dir=test_dataset_root))
