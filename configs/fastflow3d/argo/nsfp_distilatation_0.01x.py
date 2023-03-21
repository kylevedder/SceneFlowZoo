_base_ = "./nsfp_distilatation.py"

epochs = 5000
dataset = dict(args=dict(subset_fraction=0.01))