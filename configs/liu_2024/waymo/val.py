is_trainable = False
has_labels = False

test_dataset_root = "/efs/waymo_open_processed_flow/validation/"
save_output_folder = "/efs/waymo_open_processed_flow/val_liu_et_al/"

SEQUENCE_LENGTH = 3

epochs = 20
learning_rate = 2e-6
save_every = 500
validate_every = 500


model = dict(
    name="Liu2024OptimizationLoop",
    args=dict(),
)

test_dataset = dict(
    name="TorchFullFrameDataset",
    args=dict(
        dataset_name="WaymoOpenCausalSceneFlow",
        root_dir=test_dataset_root,
        with_rgb=False,
        eval_type="bucketed_epe",
        max_pc_points=180000,
        allow_pc_slicing=True,
        eval_args=dict(output_path="eval_results/bucketed_epe/waymo/liu_et_al/"),
        subsequence_length=SEQUENCE_LENGTH,
    ),
)


test_dataloader = dict(args=dict(batch_size=1, num_workers=0, shuffle=False, pin_memory=True))
