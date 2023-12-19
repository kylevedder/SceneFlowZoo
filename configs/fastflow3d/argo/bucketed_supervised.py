_base_ = "./bucketed_nsfp_distillation_1x.py"

train_dataset = dict(args=dict(use_gt_flow=True))
test_dataset = dict(
    args=dict(
              eval_args=dict(output_path = "/tmp/frame_results/bucketed_epe/supervised/")))