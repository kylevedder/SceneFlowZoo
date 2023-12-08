_base_ = "./bucketed_flow_test_av2_class_scaled.py"

test_dataset = dict(
    args=dict(eval_type="raw_epe", eval_args=dict(_delete_=True)))
