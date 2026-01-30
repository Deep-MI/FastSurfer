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
#   The base image to build the venv and freesurfer build images from
#   - default: ubuntu:22.04
# - RUNTIME_BASE_IMAGE:
#   The base image to build the runtime image on.
#   - default: ubuntu:22.04
# - FREESURFER_BUILD_IMAGE:
#   Image to use to install freesurfer binaries from, the freesurfer binaries should be located in /opt/freesurfer in
#   the image.
#   - default: build_freesurfer
# - VENV_BUILD_IMAGE:
#   Image to use to install the python environment from, the python environment should be in /venv/ in the image.
#   - default: build_cuda
# - BUILDKIT_SBOM_SCAN_CONTEXT:
#   Enables buildkit to provide and scan build images this is active by default to provide proper SBOM manifests,
#   however, it may also include parts that are not part of the distributed image (specifically build image parts
#   installed in the build image, but not transferred to the runtime image such as git, wget, the miniconda installer,
#   etc.)
#   - default: true
# - UV_VERSION:
#   The version of uv to use to build the python environment.
#   - default: 0.9.22
# - PYTHON_VERSION:
#   The python version to use for the virtual environment.
#   - default: 3.12
# - FREESURFER_VERSION:
#   The freesurfer version used in the image.
#   - default: 7.4.1
# - FREESURFER_URL:
#   The url to download the freesurfer install archive from, if "default", the
#   - default: default
# - FASTSURFER_VERSION:
#   The fastsurfer version to include in image labels.
#   - default: dev
# - GIT_HASH:
#   The git hash of the fastsurfer version to include in image labels.
#   - default: ""
# - REPOSITORY_URL:
#   The repository URL of the fastsurfer source code to include in image labels.
#   - default: "<no repository set>"
# - DOC_URL:
#   The documentation URL of fastsurfer to include in image labels.
#   - default: "https://fastsurfer.org/fastsurfer/dev"
# - VENDOR:
#   The vendor string to include in image labels.
#   - default: "Image Analysis Lab, DZNE https://deep-mi.org/"
# - AUTHOR:
#   The author string to include in image labels.
#   - default: "David Kügler <david.kuegler@dzne.de>"
# - DEVICE:
#   The device type to build the python environment for, this affects the installed torch packages.
#   Supported values are what is supported by uv, e.g.: cu128, xpu, rocm6.2.4, etc.
#   - default: cu128
# - DEBUG:
#   If set to "true", retains build debug output during build.
#   - default: false
# - INSECURE_FLAG:
#   If set to "--insecure", will download freesurfer without verifying the ssl certificate.
#   - default: ""

# DOCUMENTATION FOR TARGETS (use '--target <VALUE>'):
# To select which imaged will be tagged with '-t'
# - runtime:
#   Build the "distributable" image, this is the "final" fastsurfer docker image.
# - build_freesurfer:
#   Build the freesurfer build image only.
# - build_common:
#   Build the basic image with the python environment (hardware/driver-agnostic)
# - build_venv:
#   Build the python environment image with cuda/rocm/cpu support

ARG BUILD_BASE_IMAGE="ubuntu:24.04"
ARG RUNTIME_BASE_IMAGE="ubuntu:24.04"
ARG VENV_BUILD_IMAGE="build_venv"
ARG FREESURFER_BUILD_IMAGE="build_freesurfer"
ARG UV_VERSION="0.9.22"
ARG PYTHON_VERSION="3.12"
ARG FREESURFER_VERSION="7.4.1"
ARG FREESURFER_URL="not set"
ARG FASTSURFER_VERSION="dev"
ARG GIT_HASH=""
ARG REPOSITORY_URL="<no repository set>"
ARG DOC_URL="https://fastsurfer.org/fastsurfer/dev"
ARG VENDOR="Image Analysis Lab, DZNE https://deep-mi.org/"
ARG AUTHOR="David Kügler <david.kuegler@dzne.de>"
ARG BUILDKIT_SBOM_SCAN_CONTEXT="true"
ARG DEVICE="cu128"
ARG DEBUG="false"
ARG INSECURE_FLAG=""

FROM ghcr.io/astral-sh/uv:$UV_VERSION AS selected_uv_install_image

## Start with ubuntu base to build the venv base stage
FROM $BUILD_BASE_IMAGE AS build_base

ARG PYTHON_VERSION

ENV LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-e", "-c"]

# Install packages needed for build
RUN <<EOF
apt-get update
apt-get install -qyy \
  -o APT::Install-Recommends=false \
  -o APT::Install-Suggests=false \
  aria2 \
  ca-certificates \
  file \
  upx
apt clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
EOF

# Install uv
COPY --from=selected_uv_install_image /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON=python$PYTHON_VERSION \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_NO_DEV=1 \
    UV_TOOL_BIN_DIR=/usr/local/bin

# create a stage for the common components used across different DEVICE settings
FROM build_base AS build_venv

ARG PYTHON_VERSION
ARG DEVICE
ARG AUTHOR
ARG REPOSITORY_URL
ARG DOC_URL
ARG FASTSURFER_VERSION
ARG GIT_HASH
ARG VENDOR
ARG BUILD_BASE_IMAGE

SHELL ["/bin/bash", "--login", "-e", "-c"]

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=/install/pyproject.toml <<EOF
uv venv --python ${PYTHON_VERSION} --relocatable /venv --prompt fastsurfer
# activate the virtual environment
source /venv/bin/activate
# uv pip sync only installs "exactly" the packages in the lockfile, but does not resolve dependencies
# therefore, we first compile the environment to resolve dependencies, and then sync to install both directly and
# indirectly required packages
uv pip compile --torch-backend=${DEVICE} --extra qc /install/pyproject.toml | uv pip sync --torch-backend=${DEVICE} -
EOF

LABEL org.opencontainers.image.authors="$AUTHOR" \
      org.opencontainers.image.url="$REPOSITORY_URL" \
      org.opencontainers.image.documentation="$DOC_URL/overview/docker.html" \
      org.opencontainers.image.source="${REPOSITORY_URL/tree/blob}/tools/Docker/Dockerfile" \
      org.opencontainers.image.version="$FASTSURFER_VERSION" \
      org.opencontainers.image.revision="$GIT_HASH" \
      org.opencontainers.image.vendor="$VENDOR" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.ref.name="deepmi/fastsurfer-build:venv-${DEVICE}_v$FASTSURFER_VERSION" \
      org.opencontainers.image.title="Virtual environment build image for FastSurfer $FASTSURFER_VERSION with $DEVICE" \
      org.opencontainers.image.description="This image contains the python environment for the official FastSurfer docker image. The corresponding virtual environment is placed at /venv in the image." \
      org.opencontainers.image.base.name="$BUILD_BASE_IMAGE"

# create a stage for pruned Freesurfer
FROM build_base AS build_freesurfer

ARG DEVICE
ARG AUTHOR
ARG REPOSITORY_URL
ARG DOC_URL
ARG FREESURFER_VERSION
ARG FREESURFER_URL
ARG FASTSURFER_VERSION
ARG GIT_HASH
ARG VENDOR
ARG BUILD_BASE_IMAGE

LABEL org.opencontainers.image.authors="$AUTHOR" \
      org.opencontainers.image.url="$REPOSITORY_URL" \
      org.opencontainers.image.documentation="$DOC_URL/overview/docker.html" \
      org.opencontainers.image.source="${REPOSITORY_URL/tree/blob}/tools/Docker/Dockerfile" \
      org.opencontainers.image.version="$FREESURFER_VERSION" \
      org.opencontainers.image.revision="$GIT_HASH" \
      org.opencontainers.image.vendor="$VENDOR" \
      org.opencontainers.image.licenses="FreeSurfer License" \
      org.opencontainers.image.ref.name="deepmi/fastsurfer-build:freesurfer${FREESURFER_VERSION//./}" \
      org.opencontainers.image.title="FreeSurfer $FREESURFER_VERSION minimal installation for FastSurfer $FASTSURFER_VERSION" \
      org.opencontainers.image.description="This image contains the FreeSurfer binaries used in FastSurfer, for example for surface reconstruction, for the official FastSurfer docker image. These binaries are placed at /opt/freesurfer in the image." \
      org.opencontainers.image.base.name="$BUILD_BASE_IMAGE"

# install freesurfer and point to new python location
RUN --mount=type=bind,source=tools/build/install_fs_pruned.sh,target=/install/install_fs_pruned.sh \
    --mount=type=bind,source=tools/build/link_fs.sh,target=/install/link_fs.sh \
    --mount=type=cache,target=/install/download \
    --mount=type=bind,source=pyproject.toml,target=/install/pyproject.toml <<EOF
# the INSECURE_FLAG build arg may be set to "--insecure" to skip ssl certificate verification during download
# for production environments, this should never be used. This option is only provided to deal with an outdated
# ssl certificate on the freesurfer download server.
if [[ -n "$INSECURE_FLAG" ]]; then
    echo "SECURITY NOTICE: Downloading freesurfer for the build image without ssl validation."
fi
/install/install_fs_pruned.sh /opt --upx --url $FREESURFER_URL $INSECURE_FLAG
/install/link_fs.sh /opt/freesurfer
EOF

COPY tools/build/fspython /opt/freesurfer/bin/fspython

# =======================================================
# Here, we create references to the requested build image
# =======================================================
# This is needed because COPY --from=<image/stage> does not accept variables as part of the image/stage name
# selected_freesurfer_build_image -> $FREESURFER_BUILD_IMAGE
FROM $FREESURFER_BUILD_IMAGE AS selected_freesurfer_build_image
# selected_venv_build_image -> $VENV_BUILD_IMAGE
FROM $VENV_BUILD_IMAGE AS selected_venv_build_image

# =========================================
# Here, we create the smaller runtime image
# =========================================
FROM $RUNTIME_BASE_IMAGE AS runtime

ARG AUTHOR
ARG REPOSITORY_URL
ARG DOC_URL
ARG FASTSURFER_VERSION
ARG GIT_HASH
ARG VENDOR
ARG DEVICE
ARG RUNTIME_BASE_IMAGE
ARG PYTHON_VERSION


LABEL org.opencontainers.image.authors="$AUTHOR" \
      org.opencontainers.image.url="$REPOSITORY_URL" \
      org.opencontainers.image.documentation="$DOC_URL/overview/docker.html" \
      org.opencontainers.image.source="${REPOSITORY_URL/tree/blob}/tools/Docker/Dockerfile" \
      org.opencontainers.image.version="$FASTSURFER_VERSION" \
      org.opencontainers.image.revision="$GIT_HASH" \
      org.opencontainers.image.vendor="$VENDOR" \
      org.opencontainers.image.licenses="FreeSurfer License" \
      org.opencontainers.image.ref.name="deepmi/fastsurfer:${DEVICE}_v${FASTSURFER_VERSION}" \
      org.opencontainers.image.title="FastSurfer $FASTSURFER_VERSION minimal installation for $DEVICE" \
      org.opencontainers.image.description="This image contains the python virtual environment for the official FastSurfer docker image. The corresponding virtual environment is placed at /venv in the image." \
      org.opencontainers.image.base.name="$RUNTIME_BASE_IMAGE"

ENV LANG=C.UTF-8 \
    LC_NUMERIC=en_US.UTF-8 \
    MPLCONFIGDIR=/tmp/matplotlib-config

SHELL ["/bin/bash", "-e", "-c"]

RUN <<EOF
# Install required packages for freesurfer to runtime
apt-get update
shell_deps="time bc"
freesurfer_deps="gawk libgomp1 libquadmath0 tcsh"
whippersnappy_opengl_deps="xvfb  libglib2.0-0 libxkbcommon-x11-0 libgl1 libegl1 libfontconfig1 libdbus-1-3"
apt-get install -y --no-install-recommends \
    $shell_deps \
    $freesurfer_deps \
    $whippersnappy_opengl_deps
apt clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# create matplotlib config dir (in case someone starts this interactively)
mkdir -m 777 $MPLCONFIGDIR
# make sure we use bash and activate venv
echo "source /venv/bin/activate" >> /etc/bash.bashrc
EOF

# Add FreeSurfer and python Environment variables
# DO_NOT_SEARCH_FS_LICENSE_IN_FREESURFER_HOME=true deactivates the search for FS_LICENSE in FREESURFER_HOME
ENV OS=Linux \
    FS_OVERRIDE=0 \
    FIX_VERTEX_AREA="" \
    SUBJECTS_DIR=/opt/freesurfer/subjects \
    FSF_OUTPUT_FORMAT=nii.gz \
    FREESURFER_HOME=/opt/freesurfer \
    PYTHONUNBUFFERED=0 \
    MPLCONFIGDIR=/tmp \
    XDG_CACHE_HOME=/tmp/xdgcache \
    PATH=/venv/bin:/opt/freesurfer/bin:$PATH \
    DO_NOT_SEARCH_FS_LICENSE_IN_FREESURFER_HOME="true"

# Copy fastsurfer venv and pruned freesurfer from build images

# Note, since COPY does not support variables in the --from parameter, so we point to a reference here, and the
# seletced_<name>_build_image is a only a reference to $<NAME>_BUILD_IMAGE
COPY --from=selected_freesurfer_build_image /opt/freesurfer /opt/freesurfer
COPY --from=selected_venv_build_image /venv /venv
# also copy the python installation to /opt/python from venv image
COPY --from=selected_venv_build_image /opt/python /opt/python

# Copy fastsurfer over from the build context and add PYTHONPATH
COPY . /fastsurfer/
ENV PYTHONPATH=/fastsurfer:/opt/freesurfer/python/packages \
    FASTSURFER_HOME=/fastsurfer \
    PATH=/fastsurfer:$PATH

RUN <<EOF
source /venv/bin/activate
cd /fastsurfer
# Download all remote network checkpoints already
python3 FastSurferCNN/download_checkpoints.py --all
# compile all FastSurfer scripts into bytecode
python3 -m compileall *
# update the build file with checkpoints md5sums and pip packages.
python3 FastSurferCNN/version.py --sections +git+checkpoints+pip --build_cache tools/Docker/BUILD.info -o BUILD.info
EOF

SHELL ["/bin/bash", "--login", "-c"]

# Set FastSurfer workdir and entrypoint
#  the script entrypoint ensures that our venv is active
USER nonroot
WORKDIR "/fastsurfer"
ENTRYPOINT ["/fastsurfer/tools/Docker/entrypoint.sh","/fastsurfer/run_fastsurfer.sh"]
CMD ["--help"]

FROM runtime AS runtime_cuda
# to support AWS docker images, see Issue #352
# https://sarus.readthedocs.io/en/stable/user/custom-cuda-images.html#controlling-the-nvidia-container-runtime

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=8.0"
