_base_ = "./val_debug.py"

model = dict(
    args=dict(
        minibatch_size=5,
        lr=0.008,
    )
)
