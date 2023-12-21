_base_ = "./bucketed_nsfp_distillation_1x.py"
test_dataset = dict(
    args=dict(
              eval_args=dict(output_path = "/tmp/frame_results/bucketed_epe/nsfp_distillation_1x_debug/")))

train_dataloader = dict(
    args=dict(batch_size=1, num_workers=0, shuffle=True, pin_memory=False))

test_dataloader = dict(
    args=dict(batch_size=1, num_workers=0, shuffle=False, pin_memory=True))