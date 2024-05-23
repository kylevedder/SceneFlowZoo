is_trainable = False
has_labels = False

test_dataset_root = "/efs/argoverse2_small/val/"
save_output_folder = "/efs/argoverse2_small/val_nsfp_rewritten_flow_debug/"


SEQUENCE_LENGTH = 2

model = dict(
    name="WholeBatchOptimizationLoop",
    args=dict(model_class="WholeBatchNSFPCycleConsistency", save_flow_every=10),
)

test_dataset = dict(
    name="TorchFullFrameDataset",
    args=dict(
        dataset_name="Argoverse2CausalSceneFlow",
        root_dir=test_dataset_root,
        with_ground=False,
        with_rgb=False,
        eval_type="bucketed_epe",
        eval_args=dict(),
    ),
)

test_dataloader = dict(args=dict(batch_size=1, num_workers=0, shuffle=False, pin_memory=True))
