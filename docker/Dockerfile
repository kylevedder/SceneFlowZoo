FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04
SHELL ["/bin/bash", "-c"]
# Set the timezone info because otherwise tzinfo blocks install 
# flow and ignores the non-interactive frontend command 🤬🤬🤬
RUN ln -snf /usr/share/zoneinfo/America/New_York /etc/localtime && echo "/usr/share/zoneinfo/America/New_York" > /etc/timezone

# Core system packages
RUN apt-get update --fix-missing
RUN apt install -y software-properties-common wget curl gpg gcc git make

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

RUN apt install -y apt-utils

ENV TORCH_CUDA_ARCH_LIST="Ampere;Turing;Pascal"
ENV FORCE_CUDA="1"
RUN conda update -y conda
RUN conda install numpy python=3.10 pytorch==1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu113_pyt1121/download.html
RUN pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
RUN pip install pyarrow pyquaternion open3d
RUN pip install tensorboard
RUN pip install pytorch-lightning==1.9.4
RUN pip install nntime
RUN pip install imageio
# RUN pip install bucketed-scene-flow-eval==0.0.3

ENV PYTHONPATH=/project:/bucketed_scene_flow_eval:/:${PYTHONPATH}
COPY bashrc /root/.bashrc
WORKDIR /project