_base_ = "./unsupervised_flow_data.py"

flow_save_folder = "/efs/argoverse2/val_nsfp_flow/"
model = dict(name="NSFPCached", args=dict(flow_save_folder=flow_save_folder))