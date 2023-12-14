_base_ = "./bucketed_nsfp_distillation.py"

train_dataset = dict(args=dict(use_gt_flow=True))
