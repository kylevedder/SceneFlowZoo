_base_ = "./val.py"

save_output_folder = "/efs/waymo_open_processed_flow/val_nsfp_debug/"

model = dict(
    name="NSFPCycleConsistencyOptimizationLoop",
    args=dict(epochs=40),
)
