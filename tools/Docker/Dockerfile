# Copyright 2023 DeepMI Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
#   The python version to use for the virtual environment. build.py always passes this explicitly
#   from tool.python.version in pyproject.toml, which is the single source of truth; the ARG
#   default only applies to a direct `docker build` and is checked against that key by
#   test/lint/test_python_version.py.
#   - default: see tool.python.version in pyproject.toml
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
# - SOURCE_URL:
#   The same repository URL in the form that links to a file rather than a directory, used for the
#   image source label. Built by build.py, because rewriting it here would replace the first match
#   in the string rather than the path segment.
#   - default: "<no repository set>"
# - DOC_URL:
#   The documentation URL of fastsurfer to include in image labels.
#   - default: "https://deep-mi.org/FastSurfer/dev"
# - VENDOR:
#   The vendor string to include in image labels.
#   - default: "DeepMI Lab, DZNE https://deep-mi.org/"
# - AUTHOR:
#   The author string to include in image labels.
#   - default: "FastSurfer Developers <https://github.com/Deep-MI/FastSurfer>"
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
ARG SOURCE_URL="<no repository set>"
ARG DOC_URL="https://deep-mi.org/FastSurfer/dev"
ARG VENDOR="DeepMI Lab, DZNE https://deep-mi.org/"
ARG AUTHOR="FastSurfer Developers <https://github.com/Deep-MI/FastSurfer>"
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
  ca-certificates
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
ARG PINNED_REQUIREMENTS=false

SHELL ["/bin/bash", "--login", "-e", "-c"]

# Install the project's dependencies using either pyproject.toml or the backend-neutral pinned requirements.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=/install/pyproject.toml \
    --mount=type=bind,source=requirements.txt,target=/install/requirements.txt <<EOF
uv venv --python ${PYTHON_VERSION} --relocatable /venv --prompt fastsurfer
# activate the virtual environment
source /venv/bin/activate
# uv pip sync only installs "exactly" the packages in the lockfile, but does not resolve dependencies
# therefore, we first compile the environment to resolve dependencies, and then sync to install both directly and
# indirectly required packages
if [[ "$PINNED_REQUIREMENTS" == "true" ]]; then
    uv pip compile --torch-backend=${DEVICE} /install/requirements.txt | uv pip sync --torch-backend=${DEVICE} -
else
    uv pip compile --torch-backend=${DEVICE} --extra container /install/pyproject.toml | uv pip sync --torch-backend=${DEVICE} -
fi
EOF

# The labels go last, and so do the args they read, see the runtime stage.
ARG AUTHOR
ARG REPOSITORY_URL
ARG SOURCE_URL
ARG DOC_URL
ARG FASTSURFER_VERSION
ARG GIT_HASH
ARG VENDOR
ARG BUILD_BASE_IMAGE

LABEL org.opencontainers.image.authors="$AUTHOR" \
      org.opencontainers.image.url="$REPOSITORY_URL" \
      org.opencontainers.image.documentation="$DOC_URL/overview/docker.html" \
      org.opencontainers.image.source="$SOURCE_URL/tools/Docker/Dockerfile" \
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
ARG FREESURFER_URL
# redeclared, because a global arg is not in scope inside a stage: without this the RUN below
# expands it to empty and --insecure never reaches the installer
ARG INSECURE_FLAG

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
/install/install_fs_pruned.sh /opt --url $FREESURFER_URL $INSECURE_FLAG
/install/link_fs.sh /opt/freesurfer
EOF

COPY tools/build/fspython /opt/freesurfer/bin/fspython

# The labels go last, and so do the args they read. A label writes metadata and no filesystem
# layer, but everything after it in the stage keys on it, so a version or a hash that changes with
# every commit would rebuild the steps above from here.
ARG AUTHOR
ARG REPOSITORY_URL
ARG SOURCE_URL
ARG DOC_URL
ARG FREESURFER_VERSION
ARG FASTSURFER_VERSION
ARG GIT_HASH
ARG VENDOR
ARG BUILD_BASE_IMAGE

LABEL org.opencontainers.image.authors="$AUTHOR" \
      org.opencontainers.image.url="$REPOSITORY_URL" \
      org.opencontainers.image.documentation="$DOC_URL/overview/docker.html" \
      org.opencontainers.image.source="$SOURCE_URL/tools/Docker/Dockerfile" \
      org.opencontainers.image.version="$FREESURFER_VERSION" \
      org.opencontainers.image.revision="$GIT_HASH" \
      org.opencontainers.image.vendor="$VENDOR" \
      org.opencontainers.image.licenses="FreeSurfer License" \
      org.opencontainers.image.ref.name="deepmi/fastsurfer-build:freesurfer${FREESURFER_VERSION//./}" \
      org.opencontainers.image.title="FreeSurfer $FREESURFER_VERSION minimal installation for FastSurfer $FASTSURFER_VERSION" \
      org.opencontainers.image.description="This image contains the FreeSurfer binaries used in FastSurfer, for example for surface reconstruction, for the official FastSurfer docker image. These binaries are placed at /opt/freesurfer in the image." \
      org.opencontainers.image.base.name="$BUILD_BASE_IMAGE"

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

ENV MPLCONFIGDIR=/tmp/matplotlib-config

SHELL ["/bin/bash", "-e", "-c"]

RUN <<EOF
# Install required packages for freesurfer to runtime
apt-get update
shell_deps="time bc"
freesurfer_deps="gawk libgomp1 libquadmath0 tcsh"
whippersnappy_egl_deps="libegl1 libgl1 libfontconfig1"
apt-get install -y --no-install-recommends \
    $shell_deps \
    $freesurfer_deps \
    $whippersnappy_egl_deps \
    locales
locale-gen en_US.UTF-8 C.UTF-8
update-locale LANG=C.UTF-8 LC_NUMERIC=en_US.UTF-8
apt clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# create matplotlib config dir (in case someone starts this interactively)
mkdir -m 777 $MPLCONFIGDIR
# make sure we use bash and activate venv
echo "source /venv/bin/activate" >> /etc/bash.bashrc
EOF

# Add FreeSurfer and python Environment variables
# DO_NOT_SEARCH_FS_LICENSE_IN_FREESURFER_HOME=true deactivates the search for FS_LICENSE in FREESURFER_HOME
# XDG_DATA_HOME pins neurolit platformdirs data files to a deterministic,
# runtime-writable image path outside the temporary directories.
# FASTSURFER_DOCKER_DEFAULT_USER identifies the image user used when docker is
# run without an explicit --user mapping.
ENV OS=Linux \
    FS_OVERRIDE=0 \
    FIX_VERTEX_AREA="" \
    SUBJECTS_DIR=/opt/freesurfer/subjects \
    FSF_OUTPUT_FORMAT=nii.gz \
    FREESURFER_HOME=/opt/freesurfer \
    PYTHONUNBUFFERED=0 \
    XDG_CACHE_HOME=/tmp/xdgcache \
    LANG=C.UTF-8 \
    LC_NUMERIC=en_US.UTF-8 \
    PATH=/venv/bin:/opt/freesurfer/bin:$PATH \
    MPLCONFIGDIR=/tmp/matplotlib-config \
    XDG_DATA_HOME=/var/.local/share \
    FASTSURFER_DOCKER_DEFAULT_USER=nonroot \
    DO_NOT_SEARCH_FS_LICENSE_IN_FREESURFER_HOME="true"

# Copy fastsurfer venv and pruned freesurfer from build images

# Note, since COPY does not support variables in the --from parameter, so we point to a reference here, and the
# seletced_<name>_build_image is a only a reference to $<NAME>_BUILD_IMAGE
COPY --from=selected_freesurfer_build_image /opt/freesurfer /opt/freesurfer
COPY --from=selected_venv_build_image /venv /venv
# also copy the python installation to /opt/python from venv image
COPY --from=selected_venv_build_image /opt/python /opt/python

ENV PYTHONPATH=/fastsurfer:/opt/freesurfer/python/packages \
    FASTSURFER_HOME=/fastsurfer \
    PATH=/fastsurfer:$PATH

# The downloader and what it reads, copied before the rest of FastSurfer so that the weights layer
# below keys on these files alone. With `COPY . /fastsurfer/` above it instead, every source change
# invalidated the layer and re-fetched about 700 MB, which is both slow and the step that has
# failed on a dropped connection. get_config_file resolves each module's checkpoint_paths.yaml by
# path rather than importing it, which is why the four are listed rather than their packages.
# test/lint/test_dockerfile_download_layer.py keeps this list covering the downloader's imports.
COPY FastSurferCNN/__init__.py FastSurferCNN/download_checkpoints.py /fastsurfer/FastSurferCNN/
COPY FastSurferCNN/utils /fastsurfer/FastSurferCNN/utils
COPY FastSurferCNN/config/checkpoint_paths.yaml /fastsurfer/FastSurferCNN/config/
COPY CerebNet/config/checkpoint_paths.yaml /fastsurfer/CerebNet/config/
COPY CorpusCallosum/config/checkpoint_paths.yaml /fastsurfer/CorpusCallosum/config/
COPY HypVINN/config/checkpoint_paths.yaml /fastsurfer/HypVINN/config/

RUN <<EOF
source /venv/bin/activate
cd /fastsurfer
install -d -m 0777 /var/.local "$XDG_DATA_HOME"
# TQDM_DISABLE, because a progress bar in a build log is thousands of lines that say nothing once
# the build is over. Harmless if a downloader does not use tqdm.
export TQDM_DISABLE=1
# Download all remote network checkpoints already. This retries each host itself, see
# download_checkpoint in FastSurferCNN/utils/checkpoint.py.
python3 FastSurferCNN/download_checkpoints.py --all
# Download LIT checkpoints used by the lesion inpainting integration. Retried here rather than
# inside, because it is a third-party entry point with one host and no retry of its own, and it is
# the largest download in the image.
lit_attempts=3
for ((attempt = 1; attempt <= lit_attempts; attempt++)) ; do
  if lit-download-models ; then break ; fi
  if [ "$attempt" = "$lit_attempts" ] ; then
    echo "ERROR: lit-download-models failed $attempt times, giving up."
    exit 1
  fi
  echo "WARNING: lit-download-models failed, attempt $attempt of $lit_attempts, retrying in $((attempt * 15))s."
  # a partial file left behind would make a later presence check believe the model is there.
  # :? rather than a bare expansion, so an unset variable cannot make this rm -rf /LIT
  rm -rf "${XDG_DATA_HOME:?}/LIT"
  sleep $((attempt * 15))
done
EOF

# Copy fastsurfer over from the build context. /checkpoints is in .dockerignore, so this cannot
# overwrite what the layer above downloaded.
COPY . /fastsurfer/

RUN <<EOF
source /venv/bin/activate
cd /fastsurfer
# compile all FastSurfer scripts into bytecode
python3 -m compileall *
# update the build file with checkpoints md5sums and pip packages.
python3 FastSurferCNN/version.py --sections +git+checkpoints+pip --build_cache tools/Docker/BUILD.info -o BUILD.info
EOF

SHELL ["/bin/bash", "--login", "-c"]

# Set FastSurfer workdir and entrypoint
#  the script entrypoint ensures that our venv is active
RUN <<EOF
getent group nonroot > /dev/null || groupadd --system nonroot
id -u nonroot > /dev/null 2>&1 || useradd --system --gid nonroot --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin nonroot
EOF
USER nonroot
WORKDIR "/fastsurfer"
ENTRYPOINT ["/fastsurfer/tools/Docker/entrypoint.sh","/fastsurfer/run_fastsurfer.sh"]
CMD ["--help"]

# The labels go last, and so do the args they read. A label writes metadata and no filesystem
# layer, but every step after it keys on it, and the version and the hash change with every commit,
# so higher up they rebuilt this whole stage each time, the weights download included.
ARG AUTHOR
ARG REPOSITORY_URL
ARG SOURCE_URL
ARG DOC_URL
ARG FASTSURFER_VERSION
ARG GIT_HASH
ARG VENDOR
ARG DEVICE
ARG RUNTIME_BASE_IMAGE

LABEL org.opencontainers.image.authors="$AUTHOR" \
      org.opencontainers.image.url="$REPOSITORY_URL" \
      org.opencontainers.image.documentation="$DOC_URL/overview/docker.html" \
      org.opencontainers.image.source="$SOURCE_URL/tools/Docker/Dockerfile" \
      org.opencontainers.image.version="$FASTSURFER_VERSION" \
      org.opencontainers.image.revision="$GIT_HASH" \
      org.opencontainers.image.vendor="$VENDOR" \
      org.opencontainers.image.licenses="FreeSurfer License" \
      org.opencontainers.image.ref.name="deepmi/fastsurfer:${DEVICE}_v${FASTSURFER_VERSION}" \
      org.opencontainers.image.title="FastSurfer $FASTSURFER_VERSION minimal installation for $DEVICE" \
      org.opencontainers.image.description="This image contains the python virtual environment for the official FastSurfer docker image. The corresponding virtual environment is placed at /venv in the image." \
      org.opencontainers.image.base.name="$RUNTIME_BASE_IMAGE"

FROM runtime AS runtime_cuda
# parse DEVICE to extract the CUDA version, this is needed to correctly initialize NVIDIA_REQUIRE_CUDA below
# Note, here 'cuda' defaults to 12.8
# DEVICE is redeclared, because arg scope does not cross stages: without it the substitutions
# below all start from an empty string and NVIDIA_REQUIRE_CUDA ends up as "cuda>=".
ARG DEVICE
ARG CUDA_VERSION__0=${DEVICE/cu11/11.}
ARG CUDA_VERSION__1=${CUDA_VERSION__0/cu12/12.}
ARG CUDA_VERSION__2=${CUDA_VERSION__1/cu13/13.}
ARG CUDA_VERSION__3=${CUDA_VERSION__2/cu14/14.}
ARG CUDA_VERSION__4=${CUDA_VERSION__3/cuda/12.8}

# to support AWS docker images, see Issue #352
# https://sarus.readthedocs.io/en/stable/user/custom-cuda-images.html#controlling-the-nvidia-container-runtime

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    NVIDIA_REQUIRE_CUDA="cuda>=${CUDA_VERSION__4}"
