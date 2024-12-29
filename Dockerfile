FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /home/
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    vim \
    && rm -rf /var/lib/apt/lists/*
RUN pip install matplotlib