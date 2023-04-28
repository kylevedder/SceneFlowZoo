# Scene flow label generation

Use: https://github.com/nchodosh/argoverse2-sf

# Filesystem assumptions

For some containing folder, have an `argoverse2/` folder so that the downloaded files live inside

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