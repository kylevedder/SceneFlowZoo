_base_ = "./val.py"

save_output_folder = "/efs/argoverse2/val_eulerflow_sinc_feather/"

model = dict(
    name="EulerFlowSincOptimizationLoop",
)
