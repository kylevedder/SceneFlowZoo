# How to use this code

## File system assumptions

### Argoverse 2

Somewhere on disk, have an `argoverse2/` folder so that the downloaded files live inside

```
argoverse2/train
argoverse2/val
argoverse2/test
```

and generate the train and val supervision labels to

```
argoverse2/train_sceneflow
argoverse2/val_sceneflow
```


The [Argoverse 2 Scene Flow generation script](https://github.com/nchodosh/argoverse2-sf) to compute ground truth flows for both `train/` and `val/`.

### Waymo Open

Download the Scene Flow labels contributed by _Scalable Scene Flow from Point Clouds in the Real World_. We preprocess these files, both to convert them from an annoying proto file format to a standard Python format and to remove the ground points.

Do this using the `data_prep_scripts/waymo/extract_flow_and_remove_ground.py` file in the Waymo Open docker container.

## Docker Images

This project has three different docker images for different functions.

Each image has an associated convinence script. You must edit this script to modify the mount commands to point to your Argoverse 2 / Waymo Open data locations. The `-v` commands are these mount commands. As an example,

```
-v `pwd`:/project
```

runs `pwd` inside the script, getting the current directory, and ensures that it's mounted as `/project` inside the container. You must edit the `/efs/` mount, i.e.

```
-v /efs:/efs 
```

so that the source points to the containing folder of your `argoverse2/` and `waymo_open_processed_flow/` directories. As an example, on our cluster I have `~/datasets/argoverse2` and `~/datasets/waymo_open_processed_flow`, so if I were to run this on our cluster I would modify these mounts to be

```
-v $HOME/datasets:/efs
```

It's important that, once inside the docker container, the path to the Argoverse 2 dataset is `/efs/argoverse2/...` and the path to Waymo Open is `/efs/waymo_open_processed_flow/...`

### Main image: 

Built with `docker/Dockerfile` [[dockerhub](https://hub.docker.com/repository/docker/kylevedder/zeroflow)]

Convenience launch script is `./launch.sh`. Make sure that Argoverse 2 and Waymo Open preprocessed are mounted inside a folder in the container as `/efs`.

### Waymo preprocessing image:

Built with `docker/Dockerfilewaymo` [[dockerhub](https://hub.docker.com/repository/docker/kylevedder/zeroflow_waymo)]. Includes Tensorflow and other dependencies to preprocess the raw Waymo Open format and convert it to a standard format readable in the main image.

Convenience launch script is `./launch_waymo.sh`.

### AV2 challenge submission image:

Built with `docker/Dockerav2` [[dockerhub](https://hub.docker.com/repository/docker/kylevedder/zeroflow_av2)]. Based on the main image, but includes the [AV2 API](https://github.com/argoverse/av2-api).

Convenience launch script is `./launch_av2.sh`.

## Setting up the base system

The base system must have CUDA 11.3+ and [NVidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) installed.
