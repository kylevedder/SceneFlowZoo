_base_ = "./supervised.py"

epochs = 200
check_val_every_n_epoch = 1
learning_rate = 2e-8
validate_every = None
dataset = dict(args=dict(subset_fraction=0.01, subset_mode='sequential'))
