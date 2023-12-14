#!/bin/bash
xhost +
touch `pwd`/docker_history.txt
docker run --gpus=all --rm -it \
 --shm-size=16gb \
 -v `pwd`:/project \
 -v /efs:/efs \
 -v /bigdata:/bigdata \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 -v /tmp/frame_results:/tmp/frame_results \
 -v `pwd`/../bucketed_scene_flow_eval:/bucketed_scene_flow_eval \
 -v `pwd`/docker_history.txt:/root/.bash_history \
 -e DISPLAY=$DISPLAY \
 -h $HOSTNAME \
 --privileged \
 zeroflow:latest
 