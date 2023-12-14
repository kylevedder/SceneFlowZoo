_base_ = "./pseudolabel_train.py"

test_dataset_root = "/efs/argoverse2/val/"
flow_save_folder = "/efs/argoverse2/val_nsfp_flow/"

model = dict(args=dict(flow_save_folder=flow_save_folder))

test_dataset = dict(args=dict(root_dir=test_dataset_root))
