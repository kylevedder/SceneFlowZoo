_base_ = './unsupervised.py'

dataloader = dict(args=dict(batch_size=8, num_workers=8))
train_forward_args = dict(compute_symmetry_x=True, compute_symmetry_y=True)
