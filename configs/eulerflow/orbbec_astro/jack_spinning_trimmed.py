_base_ = "./jack_spinning.py"

SEQUENCE_LENGTH = 35

test_dataset_root = "/efs/orbbec_pointclouds/pointclouds-spinning-colored-trimmed/"
save_output_folder = "/efs/orbbec_pointclouds/pointclouds-spinning-colored-trimmed_flow/"

train_dataset = dict(
    name="TorchFullFrameDataset",
    args=dict(
        root_dir=test_dataset_root,
        subsequence_length=SEQUENCE_LENGTH,
    ),
)

test_dataset = train_dataset.copy()
