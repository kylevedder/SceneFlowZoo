#!/bin/bash

export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=localhost
export ROS_IP=127.0.0.1
mkdir -p /home/k/.ros
# Launch roscore under screen
screen -d -m -S roscore bash -c "roscore"
# Sleep for 1 seconds to allow roscore to start
sleep 1
rosparam set /use_sim_time true
# Launch openni_launch under screen
# roslaunch openni_launch openni.launch load_driver:=false
# screen -d -m -S openni_launch bash -c "roslaunch openni_launch openni.launch load_driver:=false"