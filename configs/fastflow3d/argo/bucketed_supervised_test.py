_base_ = "./bucketed_supervised.py"

has_labels = False

test_dataset_root = "/efs/argoverse2/test/"

test_dataset = dict(
    args=dict(
        root_dir=test_dataset_root,
        load_flow=False,
        eval_args=dict(output_path="eval_results/bucketed_epe/supervised_test/"),
    )
)
