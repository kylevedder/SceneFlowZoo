_base_ = "./val.py"

save_output_folder = "/efs/argoverse2/val_gigachad_nsf_no_cycle_loss_feather/"

model = dict(
    name="GigachadNSFNoCycleConsistencyLossOptimizationLoop",
)