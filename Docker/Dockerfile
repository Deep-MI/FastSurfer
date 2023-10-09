# Copyright 2023 Image Analysis Lab, German Center for Neurodegenerative Diseases(DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DOCUMENTATION FOR BUILD ARGS (use '--build-arg DEVICE=<VALUE>'):
# - BUILD_BASE_IMAGE:
#   The base image to build the conda and freesurfer build images from
#   - default: ubuntu:22.04
# - RUNTIME_BASE_IMAGE:
#   The base image to build the runtime image on.
#   - default: ubuntu:22.04
# - FREESURFER_BUILD_IMAGE:
#   Image to use to install freesurfer binaries from, the freesurfer binaries
#   should be located in /opt/freesurfer in the image.
#   - default: build_freesurfer
# - CONDA_BUILD_IAMGE:
#   Image to use to install the python environment from, the python environment
#   should be in /venv/ in the image.
#   - default: build_cuda
# - CONDA_FILE:
#   Which conda minifile to download to install conda
#   from https://repo.continuum.io/miniconda/${CONDA_FILE}
#   - default: Miniconda3-py38_4.11.0-Linux-x86_64.sh

# DOCUMENTATION FOR TARGETS (use '--target <VALUE>'):
# To select which imaged will be tagged with '-t'
# - runtime:
#   Build the "distributable" image, this is the "final" fastsurfer docker image.
# - build_freesurfer:
#   Build the freesurfer build image only.
# - build_common:
#   Build the basic image with the python enviroment (hardware/driver-agnostic)
# - build_conda:
#   Build the python environment image with cuda/rocm/cpu support

ARG FREESURFER_BUILD_IMAGE=build_freesurfer
ARG CONDA_BUILD_IMAGE=build_conda
ARG RUNTIME_BASE_IMAGE=ubuntu:22.04
ARG BUILD_BASE_IMAGE=ubuntu:22.04

## Start with ubuntu base to build the conda base stage
FROM $BUILD_BASE_IMAGE AS build_base

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Install packages needed for build
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      file \
      git \
      upx \
      wget && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ARG PYTHON_VERSION=3.10
ARG CONDA_FILE=Miniconda3-py310_23.3.1-0-Linux-x86_64.sh

# Install conda
RUN wget --no-check-certificate -qO ~/miniconda.sh https://repo.continuum.io/miniconda/${CONDA_FILE} && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh 

ENV PATH /opt/conda/bin:$PATH

# create a stage for the common components used across different DEVICE settings
FROM build_base AS build_common

# get install scripts into docker
COPY ./env/fastsurfer.yml ./Docker/conda_pack.sh ./Docker/install_env.py /install/

# SHELL ["/bin/bash", "--login", "-c"]
# Install conda for gpu

ARG DEBUG=false
RUN python /install/install_env.py -m base -i /install/fastsurfer.yml -o /install/base-env.yml && \
    conda env create -f "/install/base-env.yml" | tee /install/env-create.log ; \
    if [ "${DEBUG}" != "true" ]; then \
      rm /install/base-env.yml ; \
    fi

FROM build_common AS build_conda

ARG DEBUG=false
ARG DEVICE=cu118
# install additional packages for cuda/rocm/cpu
RUN python /install/install_env.py -m ${DEVICE} -i /install/fastsurfer.yml -o /install/${DEVICE}-env.yml && \
    conda env update -n "fastsurfer" -f "/install/${DEVICE}-env.yml" | tee /install/env-update.log && \
    /install/conda_pack.sh "fastsurfer" && \
    echo "DEBUG=$DEBUG\nDEVICE=$DEVICE\n" > /install/build_conda.args ;  \
    if [ "${DEBUG}" != "true" ]; then \
      conda env remove -n "fastsurfer" && \
      rm -R /install ; \
    fi

# create a stage for pruned Freesurfer
FROM build_base AS build_freesurfer

# get install scripts into docker
COPY ./Docker/install_fs_pruned.sh /install/
SHELL ["/bin/bash", "--login", "-c"]

# install freesurfer and point to new python location
RUN /install/install_fs_pruned.sh /opt --upx && \
rm /opt/freesurfer/bin/fspython &&  \
rm -R /install && \
ln -s /venv/bin/python3 /opt/freesurfer/bin/fspython


# =======================================================
# Here, we create references to the requested build image
# =======================================================
# This is needed because COPY --from=<image/stage> does not accept variables as part of the image/stage name
# selected_freesurfer_build_image -> $FREESURFER_BUILD_IMAGE
FROM $FREESURFER_BUILD_IMAGE AS selected_freesurfer_build_image
# selected_conda_build_image -> $CONDA_BUILD_IMAGE
FROM $CONDA_BUILD_IMAGE AS selected_conda_build_image


# =========================================
# Here, we create the smaller runtime image
# =========================================
FROM $RUNTIME_BASE_IMAGE AS runtime

ENV LANG=C.UTF-8

# Install required packages for freesurfer to dry_run
RUN apt-get update && apt-get install -y --no-install-recommends \
      bc \
      gawk \
      libgomp1 \
      libquadmath0 \
      time \
      tcsh && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Add FreeSurfer and python Environment variables
ENV OS=Linux \
    FS_OVERRIDE=0 \
    FIX_VERTEX_AREA="" \
    SUBJECTS_DIR=/opt/freesurfer/subjects \
    FSF_OUTPUT_FORMAT=nii.gz \
    FREESURFER_HOME=/opt/freesurfer \
    PYTHONUNBUFFERED=0 \
    MPLCONFIGDIR=/tmp \
    PATH=/venv/bin:/opt/freesurfer/bin:$PATH \
    MPLCONFIGDIR=/tmp/matplotlib-config

# create matplotlib config dir; make sure we use bash and activate conda env
#  (in case someone starts this interactively)
RUN mkdir -m 777 $MPLCONFIGDIR && \
    echo "source /venv/bin/activate" >> /etc/bash.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Copy fastsurfer venv and pruned freesurfer from build images

# Note, since COPY does not support variables in the --from parameter, so we point to a reference here, and the
# seletced_<name>_build_image is a only a reference to $<NAME>_BUILD_IMAGE
COPY --from=selected_freesurfer_build_image /opt/freesurfer /opt/freesurfer
COPY --from=selected_conda_build_image /venv /venv
COPY /Docker/python-s /venv/bin/

# Copy fastsurfer over from the build context and add PYTHONPATH
COPY . /fastsurfer/
ENV PYTHONPATH=/fastsurfer:/opt/freesurfer/python/packages:$PYTHONPATH \
    FASTSURFER_HOME=/fastsurfer

# Download all remote network checkpoints already, compile all FastSurfer scripts into bytecode and update
# the build file with checkpoints md5sums and pip packages.
RUN cd /fastsurfer ; python3 FastSurferCNN/download_checkpoints.py --all && \
    python3 -m compileall * && \
    python3 FastSurferCNN/version.py --sections +git+checkpoints+pip --build_cache BUILD.info -o fullBUILD.info && \
    mv fullBUILD.info BUILD.info

# Set FastSurfer workdir and entrypoint
#  the script entrypoint ensures that our conda env is active
WORKDIR "/fastsurfer"
ENTRYPOINT ["/fastsurfer/Docker/entrypoint.sh","/fastsurfer/run_fastsurfer.sh"]
CMD ["--help"]

FROM runtime as runtime_cuda
# to support AWS docker images, see Issue #352
# https://sarus.readthedocs.io/en/stable/user/custom-cuda-images.html#controlling-the-nvidia-container-runtime

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=8.0"