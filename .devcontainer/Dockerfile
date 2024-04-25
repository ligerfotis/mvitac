#FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04
#FROM python:3.10.14-bookworm
FROM ubuntu:22.04

COPY . /workspace
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    sudo \
    curl \
    wget \
    git \
    unzip \
    vim \
    nano \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -r requirements.txt

ARG USERNAME=mvitacdev
ARG UID=1000
ARG GID=$UID

# Create and switch to user
RUN groupadd -g $GID $USERNAME \
    && useradd -lm -u $UID -g $USERNAME -s /bin/bash $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# RUN useradd -lm -u $UID -g $USERNAME -s /bin/bash $USERNAME \
#      && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER $USERNAME

# Create workspace so that user own this directory
RUN mkdir -p /home/$USERNAME/mvitacdev_ws/src
WORKDIR /home/$USERNAME/mvitacdev_ws

CMD ["bash"]