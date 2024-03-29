_base_ = "./bucketed_nsfp_distillation_1x.py"

POINT_CLOUD_RANGE = (-51.2, -51.2, -3, 51.2, 51.2, 3)
VOXEL_SIZE = (0.1, 0.1, POINT_CLOUD_RANGE[5] - POINT_CLOUD_RANGE[2])
PSEUDO_IMAGE_DIMS = (1024, 1024)

save_every = 500 * 8
validate_every = 500 * 8

model = dict(name="FastFlow3D",
             args=dict(VOXEL_SIZE=VOXEL_SIZE,
                       PSEUDO_IMAGE_DIMS=PSEUDO_IMAGE_DIMS,
                       FEATURE_CHANNELS=64,
                       xl_backbone=True))

test_dataset = dict(args=dict(eval_args=dict(
    output_path="eval_results/bucketed_epe/nsfp_distillation_xl_1x/")))

dataloader = dict(
    args=dict(batch_size=2, num_workers=4, shuffle=True, pin_memory=True))

test_dataloader = dict(
    args=dict(batch_size=1, num_workers=4, shuffle=False, pin_memory=True))
