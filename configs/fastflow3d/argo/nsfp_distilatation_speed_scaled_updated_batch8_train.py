_base_ = "./nsfp_distilatation_speed_scaled_updated.py"

dataloader = dict(args=dict(batch_size=8, num_workers=8))

test_dataloader = dict(args=dict(batch_size=4, num_workers=4))
