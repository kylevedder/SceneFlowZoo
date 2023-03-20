_base_ = "./supervised.py"
epochs = 100
loss_fn = dict(name="FastFlow3DSupervisedLoss", args=dict(scale_background=False))
