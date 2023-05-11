_base_ = "./supervised.py"

epochs = 300
check_val_every_n_epoch = 1
learning_rate = 2e-8
validate_every = None
dataset = dict(args=dict(subset_fraction=0.5, subset_mode='sequential'))