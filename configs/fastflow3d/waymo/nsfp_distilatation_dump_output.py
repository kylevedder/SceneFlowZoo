_base_ = "./nsfp_distilatation.py"

save_output_folder = "/bigdata/waymo_open_processed_flow/val_distillation_out/"

test_dataloader = dict(
    args=dict(batch_size=16, num_workers=16, shuffle=False, pin_memory=True))