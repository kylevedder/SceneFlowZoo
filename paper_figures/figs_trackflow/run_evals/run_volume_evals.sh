#!/bin/bash
for f in /efs2/argoverse2_submission_evals/*; do
    echo "Processing $f file..."
    # Extract the full filename without the path
    filename=$(basename -- "$f")
    python ~/code/bucketed_scene_flow_eval/scripts/evals/av2_eval.py /efs/argoverse2/test/ /efs/argoverse2/test_sceneflow_volume_feather/ "/efs2/argoverse2_submission_evals/$filename/submission" "/efs2/argoverse2_submission_evals/$filename/volume_eval_results" --cache_root /tmp/av2_evals --cpu_count 46 --eval_type bucketed_volume_epe
done
