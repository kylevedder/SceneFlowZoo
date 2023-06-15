# ZeroFlow: Fast Zero Label Scene Flow via Distillation

[Kyle Vedder](http://vedder.io), [Neehar Peri](http://www.neeharperi.com/), [Nathaniel Chodosh](https://scholar.google.com/citations?user=b4qKr7gAAAAJ&hl=en), [Ishan Khatri](https://ishan.khatri.io/), [Eric Eaton](https://www.seas.upenn.edu/~eeaton/), [Dinesh Jayaraman](https://www.seas.upenn.edu/~dineshj/), [Yang Liu](https://youngleox.github.io/), [Deva Ramanan](https://www.cs.cmu.edu/~deva/), and [James Hays](https://faculty.cc.gatech.edu/~hays/)

Project webpage: [vedder.io/zeroflow](http://vedder.io/zeroflow)

arXiv link: [arxiv.org/abs/2305.10424](http://arxiv.org/abs/2305.10424)

**Citation:**

```
@article{Vedder2023zeroflow,
    author    = {Kyle Vedder and Neehar Peri and Nathaniel Chodosh and Ishan Khatri and Eric Eaton and Dinesh Jayaraman and Yang Liu Deva Ramanan and James Hays},
    title     = {{ZeroFlow: Fast Zero Label Scene Flow via Distillation}},
    journal   = {arXiv},
    year      = {2023},
}
```

# Docker Images

This project has three different docker images for different functions.

## Main image: 

Built with `docker/Dockerfile` [[dockerhub](https://hub.docker.com/repository/docker/kylevedder/zeroflow)]/

## Waymo preprocessing image:

Built with `docker/Dockerfilewaymo` [[dockerhub](https://hub.docker.com/repository/docker/kylevedder/zeroflow_waymo)]. Includes Tensorflow and other dependencies to preprocess the raw Waymo Open format and convert it to a standard format readable in the main image.

## AV2 challenge submission image:

Built with `docker/Dockerav2` [[dockerhub](https://hub.docker.com/repository/docker/kylevedder/zeroflow_av2)]. Based on the main image, but includes the [AV2 API](https://github.com/argoverse/av2-api).

# File system assumptions

## Argoverse 2

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

## Waymo Open

Download the Scene Flow labels contributed by _Scalable Scene Flow from Point Clouds in the Real World_. We preprocess these files, both to convert them from an annoying proto file format to a standard Python format and to remove the ground points.

Do this using the `data_prep_scripts/waymo/extract_flow_and_remove_ground.py` file.

Preprocess these files, run 

# Install Nvidia Docker 

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

# Pulling and running the Docker image

The docker image lives prebuilt on Dockerhub, at https://hub.docker.com/repository/docker/kylevedder/offline_sceneflow/general.

I have a convenience launch script to download this image and launch everything inside `launch.sh`. This *must* be run from the root of the repo. It will mount the repo and the datasets folder in the proper place. 

You must edit this script to modify the mount commands to point to your Argoverse 2 install location. The `-v` commands are these mount commands. As an example,

```
-v `pwd`:/project
```

runs `pwd` inside the script, getting the current directory, and ensures that it's mounted as `/project` inside the container. You must edit the `/efs/` mount, i.e.

```
-v /efs:/efs 
```

so that the source points to the containing folder of your `argoverse2/` directory. As an example, on our cluster I have `~/datasets/argoverse2`, so if I were to run this on our cluster I would modify these mounts to be

```
-v $HOME/datasets:/efs
```

It's important that, once inside the docker container, the path to the Argoverse 2 dataset is `/efs/argoverse2/...`

# Running the supervised experiment on 8x 3090s

Once all of this is configured, launch the container with `./launch.sh`. I suggest running this inside a terminal multiplexer (`screen`, `tmux`, etc so that you can disconnect and leave the job running). Once inside, run 

```
python train_pl.py configs/fastflow3d/argo/supervised_batch8_train.py --gpus 8
```

and training should start. It will start by verifying the val dataloader works, and then launch the train job. Assuming things are sized correctly, there should be `3600` steps per epoch.

# Submitting to the AV2 Scene Flow competition

1. Dump the outputs of the model
    a. `configs/fastflow3d/argo/nsfp_distilatation_dump_output.py` to dump the `val` set result
    b. `configs/fastflow3d/argo/nsfp_distilatation_dump_output_test.py` to dump the `test` set result
2. Convert to the competition submission format (`competition_submit.py`)