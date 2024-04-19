#!/bin/bash

# Check if MY_UID is set
if [ -z "$MY_UID" ]; then
  echo "export MY_UID=$(id -u)" >> ~/.bashrc
  uid_set=1
fi

# Check if MY_GID is set
if [ -z "$MY_GID" ]; then
  echo "export MY_GID=$(id -g)" >> ~/.bashrc
  gid_set=1
fi

# If either MY_UID or MY_GID were not set
if [ "$uid_set" == "1" ] || [ "$gid_set" == "1" ]; then
  echo "MY_UID or MY_GID were not set and have been added to your .bashrc."
  echo "Please source your .bashrc file and restart Visual Studio Code."
  exit 1
fi
