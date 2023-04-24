_base_ = "./nsfp_distilatation.py"

train_flow_dir = "/efs/argoverse2/train_chodosh_flow/"

loader = dict(args=dict(flow_data_path=train_flow_dir))
