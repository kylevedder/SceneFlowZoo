_base_ = "./val.py"

test_dataset_root = "/efs/argoverse2/val/"
save_output_folder = "/tmp/argoverse2/val_fast_nsf_flow/"

test_dataset = dict(args=dict(root_dir=test_dataset_root))
