#!/bin/bash
python test_pl.py tests/constant_baseline/config_no_rgb.py --cpu
# If nonzero exit code, exit
if [ $? -ne 0 ]; then
  exit 1
fi

python test_pl.py tests/constant_baseline/config_with_rgb.py --cpu
# If nonzero exit code, exit
if [ $? -ne 0 ]; then
  exit 1
fi


pytest tests/constant_baseline/validate_output.py

# If nonzero exit code, exit
if [ $? -ne 0 ]; then
  exit 1
fi