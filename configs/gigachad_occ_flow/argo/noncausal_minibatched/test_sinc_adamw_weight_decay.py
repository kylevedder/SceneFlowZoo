_base_ = "./test_sinc.py"

save_output_folder = "/efs/argoverse2/test_gigachad_occ_flow_sinc_adamw_feather/"

model = dict(
    args=dict(
        optimizer_type="adamw",
        weight_decay=0.000001,
    ),
)
