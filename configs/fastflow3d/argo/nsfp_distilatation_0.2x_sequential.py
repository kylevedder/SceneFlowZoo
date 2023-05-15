_base_ = "./nsfp_distilatation.py"

epochs = 100
check_val_every_n_epoch = 1
validate_every = None
dataset = dict(args=dict(subset_fraction=0.2, subset_mode='sequential'))