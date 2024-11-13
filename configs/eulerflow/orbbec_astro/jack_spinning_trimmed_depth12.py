_base_ = "./jack_spinning_trimmed.py"

save_output_folder = "/efs/orbbec_pointclouds/pointclouds-spinning-colored-trimmed-depth12_flow/"

model = dict(
    name="EulerFlowDepth12OptimizationLoop",
)
