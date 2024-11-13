#!/bin/bash
mkdir -p /home/user;
cp /payload_files/bashrc /home/user/.bashrc;
ln -s /payload_files/bash_history /home/user/.bash_history;
mkdir -p /home/user/;
mkdir -p /home/user/.cache/torch/hub/;
ln -s /payload_files/cache/torch/hub/checkpoints /home/user/.cache/torch/hub/;
ln -s /payload_files/cache/torch/hub/main.zip /home/user/.cache/torch/hub/main.zip;
touch /home/user/hello_world.txt;
# If  /opt/ros/noetic/setup.bash exists, source it
if [ -f "/opt/ros/noetic/setup.bash" ]; then
  source /opt/ros/noetic/setup.bash;
fi
bash