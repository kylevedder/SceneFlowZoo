_base_ = "./bucketed_nsfp_distillation_1x.py"

train_sequence_dir = "/efs/argoverse2_tiny/val/"
test_dataset_root = "/efs/argoverse2_tiny/val/"

epochs = 1

train_dataset = dict(args=dict(use_gt_flow=True))
test_dataset = dict(
    args=dict(
              eval_args=dict(output_path = "/tmp/frame_results/bucketed_epe_tiny/supervised/")))

# Limit batch size to 8 to fit on 24GB RTX3090
train_dataloader = dict(
    args=dict(batch_size=8, num_workers=8, shuffle=True, pin_memory=False))
