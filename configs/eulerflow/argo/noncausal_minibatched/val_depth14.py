_base_ = "./val.py"

save_output_folder = "/efs/argoverse2/val_eulerflow_depth14_feather/"

model = dict(
    name="EulerFlowDepth14OptimizationLoop",
)
