#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=200
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:30:00
#SBATCH --container-mounts=../../datasets/:/efs/,./:/project
#SBATCH --container-image=/home/kvedder/code/offline_sceneflow/kylevedder_offline_sceneflow_latest.sqsh 
bash -c "python data_prep_scripts/rasterize_heightmap.py /efs/argoverse_lidar/ --num_neighbors 5 --cpus 200"