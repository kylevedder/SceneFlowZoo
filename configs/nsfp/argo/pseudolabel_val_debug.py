_base_ = "./pseudolabel_train.py"

test_dataset_root = "/efs/argoverse2/val/"
save_output_folder = "/tmp/argoverse2/val_nsfp_flow/"

test_dataset = dict(args=dict(root_dir=test_dataset_root))