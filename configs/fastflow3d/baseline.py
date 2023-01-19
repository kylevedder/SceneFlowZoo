sequence_dir = "/efs/argoverse2/train/"

if "argoverse2" in sequence_dir:
    # Sensor dataset is ~150 lidar frames
    max_sequence_length = 146
else:
    # Lidar dataset is ~300 lidar frames
    max_sequence_length = 296


epochs = 20
learning_rate = 1e-6
SAVE_EVERY = 250
SAVE_FOLDER = "/efs/fast_flow_3d_checkpoints/"

SEQUENCE_LENGTH = 2

loader = dict(name="ArgoverseSequenceLoader",
              args=dict(sequence_dir=sequence_dir))

dataloader = dict(args=dict(batch_size=16,
                            num_workers=32,
                            shuffle=True))

dataset = dict(name="SubsequenceDataset",
               args=dict(subsequence_length=SEQUENCE_LENGTH,
                        max_sequence_length=max_sequence_length,
                         origin_mode="FIRST_ENTRY"))

model = dict(name="FastFlow3D",
             args=dict(VOXEL_SIZE=(0.2, 0.2, 4),
                       PSEUDO_IMAGE_DIMS=(512, 512),
                       POINT_CLOUD_RANGE=(-51.2, -51.2, -3, 51.2, 51.2, 1),
                       MAX_POINTS_PER_VOXEL=128,
                       FEATURE_CHANNELS=32,
                       SEQUENCE_LENGTH=SEQUENCE_LENGTH))

loss_fn = dict(name="FastFlow3DLoss", args=dict())
