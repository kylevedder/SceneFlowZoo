#!/bin/bash
python test_pl.py tests/cache_test/config.py --cpu
# If nonzero exit code, exit
if [ $? -ne 0 ]; then
  exit 1
fi

pytest tests/cache_test/validate_output.py

# If nonzero exit code, exit
if [ $? -ne 0 ]; then
  exit 1
fi

before_timestamp=$(stat -c %Y /tmp/argoverse2_tiny/val_constant_baseline_cache_out/sequence_len_002/02678d04-cc9f-3148-9f95-1ba66347dff9/0000000000.feather)

python test_pl.py tests/cache_test/config.py --cpu
# If nonzero exit code, exit
if [ $? -ne 0 ]; then
  exit 1
fi


pytest tests/cache_test/validate_output.py

# If nonzero exit code, exit
if [ $? -ne 0 ]; then
  exit 1
fi

after_timestamp=$(stat -c %Y /tmp/argoverse2_tiny/val_constant_baseline_cache_out/sequence_len_002/02678d04-cc9f-3148-9f95-1ba66347dff9/0000000000.feather)

echo "Before timestamp: $before_timestamp"
echo "After timestamp: $after_timestamp"

if [ $before_timestamp -eq $after_timestamp ]; then
  echo "Timestamps are equal"
  exit 0
else
  echo "Timestamps are not equal"
  echo "Before timestamp: $before_timestamp"
  echo "After timestamp: $after_timestamp"
  exit 1
fi
