_base_ = "./unsupervised_flow_data.py"

precision = 32
flow_save_folder = "/bigdata/nsfp_results/"
model = dict(name="NSFPCached", args=dict(flow_save_folder=flow_save_folder))