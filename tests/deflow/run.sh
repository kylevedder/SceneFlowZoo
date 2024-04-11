#!/bin/bash
# Download from https://github.com/kylevedder/zeroflow_weights/raw/master/argo/supervised/supervised.ckpt 
# and save as /tmp/fastflow3d.ckpt if it does not exist

if [ ! -f /tmp/deflow.ckpt ]; then
wget https://github.com/kylevedder/zeroflow_weights/raw/master/argo/deflow/deflow_official.ckpt -O /tmp/deflow.ckpt
fi

pip install omegaconf

python test_pl.py tests/deflow/config.py --cpu --checkpoint /tmp/deflow.ckpt
