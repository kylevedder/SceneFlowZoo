_base_ = "./jack_spinning.py"

SEQUENCE_LENGTH = 74

test_dataset_root = "/efs/orbbec_pointclouds/pointclouds-nonspinning/"
save_output_folder = "/efs/orbbec_pointclouds/pointclouds-nonspinning_flow/"

test_dataset = dict(
    name="TorchFullFrameDataset",
    args=dict(
        root_dir=test_dataset_root,
        subsequence_length=SEQUENCE_LENGTH,
    ),
)
