_base_ = "./supervised_0.5x_finetune_from_base.py"
dataset = dict(args=dict(subset_mode='random'))
test_dataloader = dict(
    args=dict(batch_size=16, num_workers=16, shuffle=False, pin_memory=True))