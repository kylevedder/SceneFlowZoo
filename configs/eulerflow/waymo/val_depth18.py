_base_ = "./val.py"

save_output_folder = "/efs/waymo_open_processed_flow/val_eulerflow_depth18/"

model = dict(
    name="EulerFlowDepth18OptimizationLoop",
)
