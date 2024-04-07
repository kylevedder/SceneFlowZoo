#!/bin/bash
# Download from https://github.com/kylevedder/zeroflow_weights/raw/master/argo/supervised/supervised.ckpt 
# and save as /tmp/fastflow3d.ckpt if it does not exist

if [ ! -f /tmp/fastflow3d.ckpt ]; then
    wget https://github.com/kylevedder/zeroflow_weights/raw/master/argo/supervised/supervised.ckpt -O /tmp/fastflow3d.ckpt
fi

python test_pl.py tests/fastflow3d/config.py --cpu --checkpoint /tmp/fastflow3d.ckpt
