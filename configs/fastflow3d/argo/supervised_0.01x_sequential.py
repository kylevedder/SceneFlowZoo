_base_ = "./supervised.py"

epochs = 5000
check_val_every_n_epoch = 100
validate_every = None
dataset = dict(args=dict(subset_fraction=0.01, subset_mode='sequential'))
