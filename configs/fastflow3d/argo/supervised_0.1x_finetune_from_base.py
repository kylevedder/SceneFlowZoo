_base_ = "./supervised.py"

epochs = 100
check_val_every_n_epoch = 3
validate_every = None
dataset = dict(args=dict(subset_fraction=0.1, subset_mode='sequential'))