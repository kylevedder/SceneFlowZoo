#!/bin/bash
xhost +
touch `pwd`/docker_history_blender.txt
docker run --gpus=all --rm -it \
 --shm-size=16gb \
 -v `pwd`:/project \
  --security-opt seccomp=unconfined `#optional` \
  -e PUID=1000 \
  -e PGID=1000 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e TZ=Etc/UTC \
  -e SUBFOLDER=/ `#optional` \
  -p 3000:3000 \
  -p 3001:3001 \
  -v /path/to/config:/config \
  -v `pwd`:/project \
  -v `pwd`/docker_history_blender.txt:/root/.bash_history \
  -v /efs:/efs \
  -v /efs2:/efs2 \
  -v /bigdata:/bigdata \
  blender_render:latest bash