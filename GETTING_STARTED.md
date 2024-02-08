# How to use this code

## Model weights

Model weights for all trained methods (FastFlow3D and ZeroFlow) and their ablations are provided in their own [GitHub repo](https://github.com/kylevedder/zeroflow_weights).

## File system assumptions

To set up the file system, follow the [Getting Started](https://github.com/kylevedder/BucketedSceneFlowEval/blob/master/docs/GETTING_STARTED.md) guide for [BucketedSceneFlowEval](https://github.com/kylevedder/BucketedSceneFlowEval). This will set up the Argoverse 2 and Waymo Open datasets in the correct format.

## Docker Images

This project provides a docker image for training and evaluation using the Docerkfile in `docker/Dockerfile` [[dockerhub](https://hub.docker.com/repository/docker/kylevedder/zeroflow_bucketed)]. The launch script `./launch.sh` will start the docker container with the correct mounts and environment variables.

These mounts are

```
-v `pwd`:/project
```

runs `pwd` inside the script, getting the current directory, and ensures that it's mounted as `/project` inside the container. 

The `/efs/` mounts are for the Argoverse 2 and Waymo Open datasets. You must link Argoverse 2 so that inside the container it appears at `/efs/argoverse2` and Waymo Open so that inside the container it appears at `/efs/waymo_open_processed_flow`. If you have these datasets in a different location, you can modify the `/efs` mount in `launch.sh` to point to the correct location.


## Setting up the base system

The base system must have driver support for CUDA 11.3+ (you do not actually need CUDA installed on your base system, but you do need a driver version that supports the container install) and [NVidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) installed.
