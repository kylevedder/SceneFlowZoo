_base_ = ["./config_with_rgb.py"]

test_dataset = dict(
    args=dict(
        with_rgb=False,
    ),
)
