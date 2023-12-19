_base_ = "./bucketed_nsfp_distillation_xl_1x.py"

test_dataset = dict(args=dict(eval_args=dict(
    output_path="/tmp/frame_results/bucketed_epe/nsfp_distillation_xl_3x/")))
