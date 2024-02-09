_base_ = "./bucketed_nsfp_distillation_1x_test.py"

test_dataset = dict(
    args=dict(
              eval_args=dict(output_path = "eval_results/bucketed_epe/nsfp_distillation_3x_test/")))