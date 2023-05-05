_base_ = "./chodosh_distilatation.py"

save_output_folder = "/efs/argoverse2/val_chodosh_distilation_out/"

test_dataloader = dict(
    args=dict(batch_size=16, num_workers=16, shuffle=False, pin_memory=True))