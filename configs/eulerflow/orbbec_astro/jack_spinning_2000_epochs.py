_base_ = "./jack_spinning.py"

model = dict(
    name="EulerFlowOptimizationLoop",
    args=dict(
        epochs=2000,
    ),
)
