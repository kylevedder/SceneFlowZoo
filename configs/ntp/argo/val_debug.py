_base_ = "./val.py"


test_dataset = dict(
    args=dict(
        log_subset=["02678d04-cc9f-3148-9f95-1ba66347dff9seqlen020idx000004"], subsequence_length=20
    )
)
