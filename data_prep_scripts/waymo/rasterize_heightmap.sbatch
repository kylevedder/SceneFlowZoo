#!/bin/bash

## Resource Request
#SBATCH --job-name=waymo_rasterize
#SBATCH --output=waymo_rasterize.out
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --cpus-per-task=120
#SBATCH --mem-per-cpu=2GB
#SBATCH --container-mounts=../../datasets/:/efs/,./:/project
#SBATCH --container-image=/home/kvedder/kylevedder+offline_sceneflow_waymo+latest.sqsh

## Job Steps
python rasterize_heightmap.py /efs/waymo_open_with_maps/ /efs/waymo_open_raster_heights/ --cpus 120