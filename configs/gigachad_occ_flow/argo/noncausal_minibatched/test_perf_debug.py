_base_ = "./test.py"

save_output_folder = "/tmp/argoverse2/test_gigachad_perf/"

model = dict(
    args=dict(
        pc_distance_type="forward_only",
        epochs=1,
    ),
)


test_dataset = dict(
    args=dict(log_subset=["af8471e6-6780-3df2-bc6a-1982a4b1b437"], subsequence_length=156)
)
