#!/bin/bash
xhost +
touch `pwd`/docker_history_rosbag.txt
docker run --gpus=all --rm -it \
 --shm-size=16gb \
 -v `pwd`:/project \
 -v /efs:/efs \
 -v /efs2:/efs2 \
 -v /bigdata:/bigdata \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 -v /tmp/frame_results:/tmp/frame_results \
 -v /tmp:/tmp \
 -v /sshfs_mounts/:/sshfs_mounts \
 -v `pwd`/docker_history_rosbag.txt:/payload_files/bash_history \
 -u $(id -u):$(id -g) \
 -v /etc/passwd:/etc/passwd:ro \
 -v /etc/group:/etc/group:ro \
 -e DISPLAY=$DISPLAY \
 -h $HOSTNAME \
 rosbag_processor:latest