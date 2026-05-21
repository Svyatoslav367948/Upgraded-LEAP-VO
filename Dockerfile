FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV TORCH_CUDA_ARCH_LIST="8.6"

# System deps
RUN apt-get update && apt-get install -y \
    git wget unzip curl build-essential \
    python3-dev python3-pip \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh

# Init conda
RUN conda init bash

# Workdir
WORKDIR /workspace

# Clone repo
RUN git clone https://github.com/wrchen530/leapvo.git

WORKDIR /workspace/leapvo

#Acceptin Anaconda
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create environment
RUN conda env create -f environment.yml && \
    conda clean -afy

# Install Eigen + project inside env
RUN /bin/bash -c "source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate leapvo && \
    wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip && \
    unzip eigen-3.4.0.zip -d thirdparty && \
    pip install ."

# Auto-activate env on container start
RUN echo "source $CONDA_DIR/etc/profile.d/conda.sh && conda activate leapvo" >> ~/.bashrc

CMD ["bash"]
