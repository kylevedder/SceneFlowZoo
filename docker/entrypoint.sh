#!/bin/bash
mkdir -p /home/user;
cp /payload_files/bashrc /home/user/.bashrc;
ln -s /payload_files/bash_history /home/user/.bash_history;
touch /home/user/hello_world.txt;
bash