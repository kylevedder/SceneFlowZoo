has_labels = False

SEQUENCE_LENGTH = 69

test_dataset_root = "/efs/orbbec_pointclouds/pointclouds-spinning-colored/"
save_output_folder = "/efs/orbbec_pointclouds/pointclouds-spinning_flow/"


model = dict(
    name="GigachadNSFOptimizationLoop",
    args=dict(
        save_flow_every=30,
        minibatch_size=5,
        speed_threshold=60.0 / 10.0,
        lr=0.00008,
        epochs=1000,
        pc_target_type="lidar",
        pc_loss_type="truncated_kd_tree_forward_backward",
    ),
)

train_dataset = dict(
    name="TorchFullFrameDataset",
    args=dict(
        dataset_name="OrbbecAstra",
        root_dir=test_dataset_root,
        flow_dir=None,
        max_pc_points=307200,
        allow_pc_slicing=False,
        subsequence_length=SEQUENCE_LENGTH,
    ),
)


train_dataloader = dict(args=dict(batch_size=1, num_workers=0, shuffle=False, pin_memory=True))


test_dataset = train_dataset.copy()
test_dataloader = train_dataloader.copy()
