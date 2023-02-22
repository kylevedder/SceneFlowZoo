_base_ = "./nsfp_distillation.py"

epochs = 50
dataset = dict(args=dict(subset_fraction=0.5))
