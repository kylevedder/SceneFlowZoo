_base_ = "./supervised.py"

test_dataloader = dict(args=dict(batch_size=1))
test_loader = dict(args=dict(num_sequences=100))
test_dataset = dict(args=dict(max_sequence_length=2))
