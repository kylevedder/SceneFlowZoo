#!/bin/bash

## Resource Request
#SBATCH --job-name=waymo_rasterize
#SBATCH --output=waymo_rasterize.out
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --cpus-per-task=200
#SBATCH --mem-per-cpu=1.2G

## Job Steps
python rasterize_heightmap.py /efs/waymo_open_with_maps/ /efs/waymo_open_raster_heights/ --cpus 250