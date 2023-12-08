_base_ = "./bucketed_flow_test_av2_class_scaled.py"

test_dataset = dict(args=dict(eval_args=dict(scaling_type="4d")))

