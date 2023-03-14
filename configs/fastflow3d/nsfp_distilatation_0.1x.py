_base_ = "./nsfp_distilatation.py"

epochs = 500
validate_every = 158 # Number of training batches

dataset = dict(args=dict(subset_fraction=0.1))
