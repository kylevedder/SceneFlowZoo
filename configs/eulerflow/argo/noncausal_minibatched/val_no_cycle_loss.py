_base_ = "./val.py"

save_output_folder = "/efs/argoverse2/val_eulerflow_no_cycle_loss_feather/"

model = dict(
    name="EulerFlowNoCycleConsistencyLossOptimizationLoop",
)
