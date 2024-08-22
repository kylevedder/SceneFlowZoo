_base_ = "./supervised.py"

save_output_folder = "/efs/waymo_open_processed_flow/val_zeroflow_feather/"

test_dataset = dict(
    args=dict(
        eval_args=dict(output_path="eval_results/bucketed_epe/waymo/fastflow3d_zeroflow/"),
    ),
)
