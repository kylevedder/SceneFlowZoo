_base_ = "./supervised_0.5x_finetune_from_base.py"
learning_rate = 2e-7
dataset = dict(args=dict(subset_mode='random'))