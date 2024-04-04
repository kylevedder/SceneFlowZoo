_base_ = "./pseudolabel_train.py"

test_dataset_root = "/efs/argoverse2/test/"
save_output_folder = "/efs/argoverse2/test_nsfp_flow_feather/"

test_dataset = dict(args=dict(root_dir=test_dataset_root, load_flow=False))
