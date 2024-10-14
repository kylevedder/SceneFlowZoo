_base_ = "./flow4d_config.py"

train_sequence_dir = "/efs/argoverse2_tiny/val/"
test_dataset_root = "/efs/argoverse2_tiny/val/"

save_output_folder = "/efs/argoverse2_tiny/val_fastflow3d_rewrite_api/"

epochs = 1

train_dataset = dict(args=dict(root_dir=train_sequence_dir, use_gt_flow=True))
test_dataset = dict(
    args=dict(
        root_dir=test_dataset_root,
        eval_args=dict(output_path="eval_results/bucketed_epe_tiny/supervised_rewrite_api/"),
    )
)

# Limit batch size to 8 to fit on 24GB RTX3090
test_dataloader = dict(args=dict(batch_size=8, num_workers=0, shuffle=False, pin_memory=True))
