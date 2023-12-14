#!/bin/bash

# Iterate over the subdirectories of /efs/argoverse2/val_dataset_stats/range_75.0/

for dir in /efs/waymo_open_processed_flow/validation_dataset_stats/range_75.0/*/
do
    # Get the directory name
    dir=${dir%*/}

    # Get the last part of the directory name
    dir=${dir##*/}

    # Delete existing video if it exists
    rm "/efs/waymo_open_processed_flow/validation_dataset_stats/range_75.0/$dir/bev.mp4"

    # Make the video
    /usr/bin/ffmpeg -framerate 10 -pattern_type glob -i "/efs/waymo_open_processed_flow/validation_dataset_stats/range_75.0/$dir/animation/*.png" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p -profile:v high -crf 18 "/efs/waymo_open_processed_flow/validation_dataset_stats/range_75.0/$dir/bev.mp4"
done