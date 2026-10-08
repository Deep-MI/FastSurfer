Building FastSurfer Docker images
=================================
In `tools/Docker`, we provide a build script and a Dockerfile for developers who want to build their own Docker
images. To run our official images, see [Running FastSurfer in a container](../../doc/overview/CONTAINERS.md). The
build script supports these platforms:

* NVIDIA / CUDA (Example 1)
* CPU (Example 2)
* AMD / ROCm (experimental, Example 3)
* Intel / XPU (very experimental)

General build settings
----------------------
The build script `build.py` supports additional arguments, targets and options, see
`python tools/Docker/build.py --help`.

Besides selecting the build arguments, the build script creates the file `BUILD.info` in the FastSurfer root
directory, which FastSurfer uses to report its version (including the git hash of the source the image was built
from). The Docker build fails without this file.
With `--dry_run`, the build script prints the command instead of executing it, so you can also run
`python tools/Docker/build.py --device cuda --dry_run | bash`.

By default, the build script tags your image as `fastsurfer:<device>-v<version_tag>`, where `<version_tag>` is
`<version>_<git_hash>` (the version from pyproject.toml and the current git hash) and `<device>` is the value of
`--device` (`cuda` and `rocm` are replaced by their default versions, e.g. `{{ CUDA_DEFAULT_STRING }}`). Specify a custom tag
with `--tag <image_tag>`.

By default, the Python environment is resolved from `pyproject.toml`, which allows the latest compatible dependency
versions. To build from the backend-neutral pinned `requirements.txt` instead, add `--pinned_requirements`. The
selected `--device` is still passed to `uv --torch-backend`, so the same pinned requirements file works for the CPU
and all supported CUDA versions, and the PyTorch wheels for the backend are selected during the build.

### BuildKit
We recommend using BuildKit to build Docker images (e.g. `DOCKER_BUILDKIT=1`; `build.py` always adds this). To
install BuildKit, run
`wget -qO ~/.docker/cli-plugins/docker-buildx https://github.com/docker/buildx/releases/download/<buildx_version>/buildx-<buildx_version>.<platform>`,
for example
`wget -qO ~/.docker/cli-plugins/docker-buildx https://github.com/docker/buildx/releases/download/v0.12.1/buildx-v0.12.1.linux-amd64`.
See also https://github.com/docker/buildx#manual-download.

Example 1: Build GPU FastSurfer Image
-------------------------------------
To build your own Docker image for FastSurfer (segmentation and surface reconstruction, for NVIDIA GPUs, including
FreeSurfer), run the following command in the FastSurfer directory:

```bash
python tools/Docker/build.py --device {{ CUDA_DEFAULT_STRING }} --tag my_fastsurfer:{{ CUDA_DEFAULT_STRING }}
```

`--device {{ CUDA_DEFAULT_STRING }}` builds for CUDA {{ CUDA_DEFAULT_VERSION }}, the default, which `--device cuda`
also selects. To build for another CUDA version, pass it to `--device`, for example
`--device {{ CUDA_LEGACY_STRING }}` for older GPUs and drivers, see
[which image fits your GPU](../../doc/overview/install/LINUX.md#nvidia-gpus);
`python tools/Docker/build.py --print_supported cuda` lists the supported versions. Add `--pinned_requirements` to
use the pinned dependency versions of `requirements.txt` (see `build.py --help` for all options).

To run the analysis, use the same command as for the official image, see
[Running FastSurfer in a container](../../doc/overview/CONTAINERS.md#using-docker), with your image:
```bash
freesurfer_license=${freesurfer_license:-/path/to/your/freesurfer/license_file}
docker run --gpus all \
           -v $HOME/my_mri_data:$HOME/my_mri_data \
           -v $HOME/my_fastsurfer_analysis:$HOME/my_fastsurfer_analysis \
           -v $freesurfer_license:$freesurfer_license \
           --rm --user $(id -u):$(id -g) my_fastsurfer:{{ CUDA_DEFAULT_STRING }} \
               --fs_license $freesurfer_license \
               --t1 $HOME/my_mri_data/subjectX/t1_weighted.nii.gz \
               --sid subjectX --sd $HOME/my_fastsurfer_analysis \
               --threads 4 --3T
```

Example 2: Build CPU FastSurfer Image
-------------------------------------
To build the Docker image for FastSurfer for the CPU only, run in the FastSurfer directory:

```bash
python tools/Docker/build.py --device cpu --tag my_fastsurfer:cpu
```

Only `--device` changes, to `cpu`.

To run the analysis, use the same command as above, but without the `--gpus all` option:
```bash
freesurfer_license=${freesurfer_license:-/path/to/your/freesurfer/license_file}
docker run -v $HOME/my_mri_data:$HOME/my_mri_data \
           -v $HOME/my_fastsurfer_analysis:$HOME/my_fastsurfer_analysis \
           -v $freesurfer_license:$freesurfer_license \
           --rm --user $(id -u):$(id -g) my_fastsurfer:cpu \
               --fs_license $freesurfer_license \
               --t1 $HOME/my_mri_data/subjectX/t1_weighted.nii.gz \
               --device cpu \
               --sid subjectX --sd $HOME/my_fastsurfer_analysis \
               --threads 16 --3T
```

Without a GPU, FastSurfer runs on the CPU anyway; `--device cpu` makes that explicit.

Example 3: Experimental Build for AMD GPUs
------------------------------------------
We also release a ROCm image, see [AMD GPUs](../../doc/overview/install/LINUX.md#amd-gpus-experimental). To build
your own, note that ROCm needs a supported OS, kernel version and GPU. Install the kernel drivers on your host
(`amdgpu-install --usecase=dkms`) for the AMD image to work, following
https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html#rocm-install-quick,
https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/amdgpu-install.html#amdgpu-install-dkms and
https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html.

```bash
python tools/Docker/build.py --device {{ ROCM_DEFAULT_STRING }} --tag my_fastsurfer:{{ ROCM_DEFAULT_STRING }}
```

`--device {{ ROCM_DEFAULT_STRING }}` builds for ROCm {{ ROCM_DEFAULT_VERSION }}, the default, which `--device rocm`
also selects; `python tools/Docker/build.py --print_supported rocm` lists the supported versions.

Run the segmentation only (FastSurfer addresses AMD GPUs as `cuda` devices, so `--device cuda` or `--device cuda:0`
selects a specific GPU):

```bash
docker run --rm --security-opt seccomp=unconfined \
           --device=/dev/kfd --device=/dev/dri --group-add video \
           -v $HOME/my_mri_data:$HOME/my_mri_data \
           -v $HOME/my_fastsurfer_analysis:$HOME/my_fastsurfer_analysis \
           --user $(id -u):$(id -g) my_fastsurfer:{{ ROCM_DEFAULT_STRING }} \
               --t1 $HOME/my_mri_data/subjectX/t1_weighted.nii.gz \
               --sid subjectX --sd $HOME/my_fastsurfer_analysis \
               --seg_only
```

Unlike the official ROCm documentation (above), we also needed to add the group render with `--group-add render` (in
addition to `--group-add video`).

We tested on an AMD Radeon Pro W6600, which is
[not officially supported](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus),
but setting `HSA_OVERRIDE_GFX_VERSION=10.3.0`
[inside Docker did the trick](https://en.opensuse.org/SDB:AMD_GPGPU#Using_CUDA_code_with_ZLUDA_and_ROCm):

```bash
docker run --rm --security-opt seccomp=unconfined \
           --device=/dev/kfd --device=/dev/dri --group-add video \
           --group-add render \
           -v $HOME/my_mri_data:$HOME/my_mri_data \
           -v $HOME/my_fastsurfer_analysis:$HOME/my_fastsurfer_analysis \
           -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
           --user $(id -u):$(id -g) my_fastsurfer:{{ ROCM_DEFAULT_STRING }} \
               --t1 $HOME/my_mri_data/subjectX/t1_weighted.nii.gz \
               --sid subjectX --sd $HOME/my_fastsurfer_analysis \
               --seg_only
```

Converting an image to Apptainer
--------------------------------
To build an Apptainer (Singularity) image from your own Docker image, add
`--singularity $HOME/my_singularity_images/fastsurfer-myimage.sif` to the `build.py` call. It first builds the image
with Docker and then converts it.

To convert the local Docker image `fastsurfer:myimage` manually, run:

```bash
singularity build $HOME/my_singularity_images/fastsurfer-myimage.sif \
    docker-daemon://fastsurfer:myimage
```

If this fails with an error message like this:
```text
INFO:    Starting build...
FATAL:   While performing build: conveyor failed to get: loading image from
  docker engine: Error response from daemon: {"message":"client version 1.22
  is too old. Minimum supported API version is 1.24, please upgrade your
  client to a newer version"}
```
export the image from Docker with `docker save -o <docker_archive_path> <image_tag>`, and build the Apptainer image
from that archive with `singularity build <sif_path> docker-archive:<docker_archive_path>`.

Build docker image with attestation and provenance
--------------------------------------------------
To build a Docker image with attestation and provenance, i.e. Software Bill Of Materials (SBOM) information, several
requirements have to be met:

1. The image must be built with version v0.11+ of BuildKit (we recommend you [install BuildKit](#buildkit) independent
   of attestation).
2. You must configure a docker-container builder in buildx
   (`docker buildx create --use --bootstrap --name fastsurfer-bctx --driver docker-container`). Here, you can add
   additional configuration options such as safe registries to the builder configuration (add
   `--config /etc/buildkitd.toml`).
   ```toml
   root = "/path/to/data/for/buildkit"
   [worker.containerd]
     gckeepstorage=9000
     [[worker.containerd.gcpolicy]]
       keepBytes = 512000000
       keepDuration = 172800
       filters = [
         "type==source.local", "type==exec.cachemount",
         "type==source.git.checkout"
       ]
     [[worker.containerd.gcpolicy]]
       all = true
       keepBytes = 1024000000
   ```
3. The standard Docker image storage driver does not support attestation files, so such images cannot be tested
   locally. There are two solutions to this limitation:
   1. Push directly to the registry:
      Add `--action push` to the build script (the default is `--action load`, which loads the created image into the
      current Docker context), and add the registry name to the image name. For example
      `python tools/Docker/build.py ... --attest --action push --tag docker.io/<account>/fastsurfer:latest`.
   2. [Install the containerd image storage driver](https://docs.docker.com/storage/containerd/#enable-containerd-image-store-on-docker-engine),
      which supports attestation. To do this on Linux, make sure your Docker daemon config file
      `/etc/docker/daemon.json` includes
      ```json
      {
          "features": {
              "containerd-snapshotter": true
          }
      }
      ```
      Note that the image storage location with containerd is not defined by the Docker config file
      `/etc/docker/daemon.json`, but by the containerd config `/etc/containerd/config.toml`, which will likely not
      exist. You can [create a default config](https://github.com/containerd/containerd/blob/main/docs/getting-started.md#customizing-containerd)
      file with `containerd config default > /etc/containerd/config.toml`, and edit its `"root"` entry (default value
      `/var/lib/containerd`).
4. Finally, build the FastSurfer image with `python tools/Docker/build.py ... --attest`, which adds the additional
   flags to the Docker build command.

Building for release
--------------------
Make sure you are building on a machine with
[containerd storage and BuildKit](#build-docker-image-with-attestation-and-provenance).

```bash
# configuration
build_dir=$HOME/FastSurfer-build
# <repo>/<name> (the push needs both!)
image=deepmi/fastsurfer
# the version can be identified with: $build_dir/run_fastsurfer.sh --version
version={{ FASTSURFER_VERSION }}
# the default CUDA image, tagged as latest, and the CUDA image for older GPUs and drivers
device_for_latest={{ CUDA_STRING }}
device_legacy={{ CUDA_LEGACY_STRING }}
# if you change the FreeSurfer version, create and upload or rename the
# FreeSurfer build image below or remove the --freesurfer_build_image argument
freesurfer_version={{ FREESURFER_VERSION }}
freesurfer_image=deepmi/fastsurfer-build:freesurfer${freesurfer_version//./}
# end of config

# code
git clone --branch stable --single-branch \
    https://github.com/Deep-MI/FastSurfer $build_dir
cd $build_dir
# supported rocm versions of this checkout's build.py
rocms=($(python3 tools/Docker/build.py --print_supported rocm))
all_tags=("latest" "cpu-latest")
# build all distinct images
for dev in cpu "${rocms[@]}" $device_legacy $device_for_latest
do
  python3 tools/Docker/build.py --tag $image:$dev-v$version \
      $([[ -n "$freesurfer_image" ]] && echo "--freesurfer_build_image $freesurfer_image") \
      --attest --device $dev --pinned_requirements
  all_tags+=("$dev-v$version")
done
# labels that are just references
docker tag $image:cpu-v$version $image:cpu-latest
docker tag $image:$device_for_latest-v$version $image:latest
# push all labels
for tag in "${all_tags[@]}" ; do docker push $image:$tag ; done
```
