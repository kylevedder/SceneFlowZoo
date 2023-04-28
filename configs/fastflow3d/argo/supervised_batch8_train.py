_base_ = "./supervised.py"

dataloader = dict(args=dict(batch_size=8, num_workers=8))
