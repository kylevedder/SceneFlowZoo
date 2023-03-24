_base_ = "./nsfp_distilatation.py"

epochs = 5000
# validate_every = 15 # Number of training batches
dataset = dict(args=dict(subset_fraction=0.01))