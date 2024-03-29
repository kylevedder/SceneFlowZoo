#!/bin/bash
docker build -f docker/Dockerfile_cuda docker/ -t zeroflow_bucketed --progress=plain
docker build -f docker/Dockerfile_cpuonly docker/ -t zeroflow_bucketed_cpu --progress=plain

docker image tag zeroflow_bucketed:latest kylevedder/zeroflow_bucketed
docker image tag zeroflow_bucketed_cpu:latest kylevedder/zeroflow_bucketed_cpu

docker image push kylevedder/zeroflow_bucketed
docker image push kylevedder/zeroflow_bucketed_cpu