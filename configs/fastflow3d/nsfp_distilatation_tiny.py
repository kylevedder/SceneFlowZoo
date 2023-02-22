_base_ = "./nsfp_distilatation.py"

epochs = 100
validate_every = 2
save_every = 2
dataset = dict(args=dict(subset_fraction=0.003))
test_dataset = dict(args=dict(subset_fraction=0.003))
