_base_ = "./val.py"

save_output_folder = "/efs/argoverse2/val_eulerflow_depth6_feather/"

model = dict(
    name="EulerFlowDepth6OptimizationLoop",
)
