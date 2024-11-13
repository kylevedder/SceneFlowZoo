_base_ = "./test_depth18.py"


test_dataset = dict(
    args=dict(
        log_subset=["5f016e44-0f38-3837-9111-58ec18d1a5e6"], subsequence_length=157, use_cache=False
    )
)
