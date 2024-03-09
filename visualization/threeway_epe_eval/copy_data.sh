#!/bin/bash

# Define the source and target directories
SOURCE_DIR="/efs2/argoverse2_submission_evals"
TARGET_DIR="/home/k/code/offline_sceneflow_eval/visualization/perf_data/threeway_epe"

# Iterate over each subfolder in the source directory
for SUBFOLDER in "$SOURCE_DIR"/*; do
    # Extract the name of the subfolder
    FOLDER_NAME=$(basename "$SUBFOLDER")
    # Define the source file path
    SOURCE_FILE="${SUBFOLDER}/eval_results/per_class_results_35.json"
    # Define the target file path, appending the subfolder name for uniqueness
    TARGET_FILE="${TARGET_DIR}/${FOLDER_NAME}_per_class_results_35.json"
    # Copy the file from the source to the target directory
    cp "$SOURCE_FILE" "$TARGET_FILE"
done

echo "Copy completed."
