#!/bin/bash

## Resource Request
#SBATCH --job-name=investigate_pointclouds_waymo
#SBATCH --output=investigate_pointclouds_waymo.out
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --cpus-per-task=120
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=00:10:00
#SBATCH --container-mounts=../../datasets/:/efs/,./:/project
#SBATCH --container-image=/home/kvedder/kylevedder_offline_sceneflow_latest.sqsh
## Job Steps
python invesigate_pointclouds.py --cpus 120