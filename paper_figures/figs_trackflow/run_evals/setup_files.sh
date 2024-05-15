#!/bin/bash

for f in /efs/argoverse2/test_*_submission.zip; do
    echo "Processing $f file..."
    # Extract the full filename without the path
    filename=$(basename -- "$f")
    # Remove the .zip extension to get the stem
    stem="${filename%.zip}"
    python ~/code/bucketed_scene_flow_eval/scripts/evals/setup_sparse_user_submission.py "/efs2/argoverse2_submission_evals/$stem/" $f /efs/argoverse2/test_sceneflow_feather
done
