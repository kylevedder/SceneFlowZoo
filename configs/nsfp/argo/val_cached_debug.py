_base_ = "./val_cached.py"

test_dataset_root = "/efs/argoverse2_tiny/val/"
save_output_folder = "/tmp/argoverse2_tiny/val_nsfp_flow/"

test_dataset = dict(args=dict(root_dir=test_dataset_root))
