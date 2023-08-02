_base_ = "./supervised.py"
loss_fn = dict(name="FastFlow3DSupervisedLoss",
               args=dict(scale_background=False, scale_speed=True))
