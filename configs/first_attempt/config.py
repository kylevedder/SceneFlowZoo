sequence_dir = "/bigdata/argoverse_lidar/train/"

learning_rate = 1e-5 * 2
save_every_iter = 50
save_folder = "/bigdata/offline_sceneflow_lower_lr/"

SEQUENCE_LENGTH = 6
BATCH_SIZE = 1

NSFP_FILTER_SIZE = 64
NSFP_NUM_LAYERS = 4

loader = dict(name="ArgoverseSequenceLoader",
              args=dict(sequence_dir=sequence_dir))

dataset = dict(name="SubsequenceDataset",
               args=dict(subsequence_length=SEQUENCE_LENGTH,
                         origin_mode="FIRST_ENTRY"))

model = dict(name="JointFlow",
             args=dict(batch_size=BATCH_SIZE,
                       VOXEL_SIZE=(0.14, 0.14, 4),
                       PSEUDO_IMAGE_DIMS=(512, 512),
                       POINT_CLOUD_RANGE=(-33.28, -33.28, -3, 33.28, 33.28, 1),
                       MAX_POINTS_PER_VOXEL=128,
                       FEATURE_CHANNELS=32,
                       FILTERS_PER_BLOCK=4,
                       PYRAMID_LAYERS=1,
                       SEQUENCE_LENGTH=SEQUENCE_LENGTH,
                       NSFP_FILTER_SIZE=NSFP_FILTER_SIZE,
                       NSFP_NUM_LAYERS=NSFP_NUM_LAYERS))

loss_fn = dict(name="JointFlowLoss",
               args=dict(NSFP_FILTER_SIZE=NSFP_FILTER_SIZE,
                         NSFP_NUM_LAYERS=NSFP_NUM_LAYERS))
