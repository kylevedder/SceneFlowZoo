_base_ = "./test_gaussian.py"

save_output_folder = "/tmp/argoverse2/test_gigachad_perf/"

model = dict(
    args=dict(
        epochs=1,
        minibatch_size=4,
    ),
)


test_dataset = dict(
    args=dict(log_subset=["34c79495-dbdf-393d-bcc6-e6f92f797628"], subsequence_length=156)
)
