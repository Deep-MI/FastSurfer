ARG DEVICE=cuda
ARG FREESURFER=pruned

## Start with ubuntu base to build the conda base stage
FROM ubuntu:22.04 AS build_conda_base

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Install packages needed for build
RUN apt-get update && apt-get install -y --no-install-recommends \
      wget \
      git \
      ca-certificates \
      upx \
      file && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ARG PYTHON_VERSION=3.10
ARG CONDA_FILE=Miniconda3-Miniconda3-py310_23.3.1-0-Linux-x86_64.sh

# Install conda
RUN wget --no-check-certificate -qO ~/miniconda.sh https://repo.continuum.io/miniconda/$CONDA_FILE  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh 

ENV PATH /opt/conda/bin:$PATH


# create a stage for reconsurf (no deep learning)
FROM build_conda_base AS build_conda_none

COPY ./env/fastsurfer_reconsurf.yml ./Docker/install_env.sh /fastsurfer/

SHELL ["/bin/bash", "--login", "-c"]
RUN /fastsurfer/install_env.sh fastsurfer_reconsurf /fastsurfer/fastsurfer_reconsurf.yml && \
    rm /fastsurfer/install_env.sh

ENV DEVICE=none

# setup shell for install command below
RUN echo "source /venv/bin/activate" >> ~/.bashrc


# create a stage for the common components used across different DEVICE settings
FROM build_conda_base AS build_conda_common

# get install scripts into docker
COPY ./env/fastsurfer.yml ./Docker/install_env.sh /fastsurfer/

SHELL ["/bin/bash", "--login", "-c"]
# Install conda for gpu
RUN /fastsurfer/install_env.sh fastsurfer /fastsurfer/fastsurfer.yml &&  \
    rm /fastsurfer/install_env.sh

# setup shell for install command below
RUN echo "source /venv/bin/activate" >> ~/.bashrc


FROM build_conda_common AS build_conda_cuda
# install additional packages for cuda
RUN conda install cudatoolkit=11.3  # this should already be satisfied through dependencies

ENV DEVICE=cuda


FROM build_conda_common AS build_conda_cpu
# install additional packages for cpu
RUN conda install --channel pytorch cpuonly=1.0

ENV DEVICE=cpu


# create a reference to the common stage for amd
FROM build_conda_common AS build_conda_amd

ENV DEVICE=amd


# create a stage for pruned Freesurfer
FROM build_conda_base AS build_fs_pruned

# get install scripts into docker
COPY ./Docker/install_fs_pruned.sh /fastsurfer/
SHELL ["/bin/bash", "--login", "-c"]

# install freesurfer and point to new python location
RUN /fastsurfer/install_fs_pruned.sh /opt --upx && \
    rm /opt/freesurfer/bin/fspython && rm /fastsurfer/install_fs_pruned.sh && \
    ln -s /venv/bin/python3 /opt/freesurfer/bin/fspython


# create a stage for no Freesurfer
FROM ubuntu:22.04 AS build_fs_none

# create an empty freesurfer directory
RUN mkdir /opt/freesurfer && echo "FreeSurfer is not installed in this Docker image" >> /opt/freesurfer/NOT_INSTALLED.md

# ========================================
# Here we define base devices for different settings for DEVICE
# ========================================

FROM ubuntu:22.04 AS base_cuda
FROM ubuntu:22.04 AS base_cpu
FROM ubuntu:22.04 AS base_none

## Start the amd image with rocm/pytorch latest
FROM rocm/pytorch AS base_amd
ENV LANG=C.UTF-8
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
ENV DEVICE=amd
ARG PYTHON_VERSION=3.10
# rocm/pytorch now uses Python 3.8.13 and has an integrated conda environment


# select the correct stages for the device and freesurfer
# this is a workaround to putting the DEVICE/FREESURFER into the --from argument of the COPY command
FROM build_conda_$DEVICE AS build_conda
FROM build_fs_$FREESURFER AS build_fs


# ========================================
# Here we create the smaller runtime image
# ========================================
FROM base_$DEVICE AS runtime

# Install required packages for freesurfer to run
RUN apt-get update && apt-get install -y --no-install-recommends \
      tcsh \
      time \
      bc \
      gawk \
      libgomp1 \
      libquadmath0 && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Add FreeSurfer Environment variables
ENV OS=Linux \
    FS_OVERRIDE=0 \
    FIX_VERTEX_AREA="" \
    SUBJECTS_DIR=/opt/freesurfer/subjects \
    FSF_OUTPUT_FORMAT=nii.gz \
    FREESURFER_HOME=/opt/freesurfer \
    PYTHONUNBUFFERED=0 \
    MPLCONFIGDIR=/tmp \
    PATH=/venv/bin:/opt/freesurfer/bin:$PATH \
    PYTHONPATH=/fastsurfer:$PYTHONPATH \
    MPLCONFIGDIR=/tmp/matplotlib-config

# create matplotlib config dir; make sure we use bash and activate conda env
#  (in case someone starts this interactively)
RUN mkdir -m 777 $MPLCONFIGDIR && \
    echo "source /venv/bin/activate" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Copy venv fastsurfer and pruned freesurfer from builder
COPY --from=build_conda /venv /venv
COPY --from=build_fs /opt/freesurfer /opt/freesurfer
COPY . /fastsurfer/

# Download all remote network checkpoints already and compile all FastSurfer scripts into bytecode
ENV PYTHONPATH=/fastsurfer:$PYTHONPATH
RUN cd /fastsurfer ; \
  if [ "$DEVICE" != "none" ]; then ; python3 FastSurferCNN/download_checkpoints.py --all ; fi ; \
  python3 -m compileall *

# Set FastSurfer workdir and entrypoint
#  the script entrypoint ensures that our conda env is active
WORKDIR "/fastsurfer"
ENV FASTSURFER_HOME=/fastsurfer
ENTRYPOINT ["./Docker/entrypoint.sh","./run_fastsurfer.sh"]
CMD ["--help"]

# special settings for the surfaces only image
FROM runtime AS runtime_surf_only
ENTRYPOINT ["./Docker/entrypoint.sh","./recon_surf/recon-surf.sh"]

# special settings for the segmentation only image
FROM runtime AS runtime_seg_only
ENTRYPOINT ["./Docker/entrypoint.sh","python", "./FastSurferCNN/run_prediction.py"]

