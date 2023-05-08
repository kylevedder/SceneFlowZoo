_base_ = "./nsfp_distilatation_speed_scaled_updated.py"
test_dataloader = dict(args=dict(batch_size=16, num_workers=16))
