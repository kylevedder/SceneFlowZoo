_base_ = "./bucketed_nsfp_distillation_xl_3x.py"

test_dataset = dict(
    args=dict(eval_args=dict(output_path="eval_results/bucketed_epe/nsfp_distillation_xl_5x/"))
)
