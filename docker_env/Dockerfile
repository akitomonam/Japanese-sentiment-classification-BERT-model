# FROM nvcr.io/nvidia/pytorch:21.03-py3
FROM pytorch/pytorch:latest
LABEL maintainer="akitomonam <https://github.com/akitomonam>"
SHELL ["/bin/bash", "-c"]

WORKDIR /workspace
ENV WORKING_DIR /workspace/data

ARG PROXY
ENV http_proxy ${HTTP_PROXY}
ENV https_proxy ${HTTP_PROXY}
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ Asia/Tokyo

RUN apt update && apt -y upgrade

RUN apt -y install  vim \
    sudo \
    wget \
    curl \
    git \
    gcc \
    cmake \
    unzip \
    tmux \
    tzdata

COPY docker_env/requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt

# Create user & add to sudoers
ARG UID
ARG GID
ARG UNAME

RUN groupadd -g ${GID} ${UNAME} && \
    useradd -u ${UID} -g ${UNAME} -G sudo  -m ${UNAME} && \
    echo "${UNAME}:${UNAME}" | chpasswd && \
    echo "${UNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

ENV UID ${UID}
ENV GID ${GID}
ENV UNAME ${UNAME}


# Change Permission
COPY ./docker_env/set_volume_permission.sh /home/${UNAME}/
RUN chmod +x /home/${UNAME}/set_volume_permission.sh

RUN /home/${UNAME}/set_volume_permission.sh

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES utility,compute
