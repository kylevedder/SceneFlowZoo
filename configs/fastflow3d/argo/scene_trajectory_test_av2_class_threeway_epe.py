_base_ = "./scene_trajectory_test_av2_class_scaled.py"

test_dataset = dict(
    args=dict(eval_type="class_threeway_epe", eval_args=dict(_delete_=True)))
