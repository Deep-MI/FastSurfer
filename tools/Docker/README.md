FastSurfer Docker Support
=========================

Pull FastSurfer from Docker Hub
-------------------------------
We provide pre-built Docker images for NVIDIA GPUs, for AMD GPUs (experimental) and for CPU-only use on
[Docker Hub](https://hub.docker.com/r/deepmi/fastsurfer/tags). To get the latest Docker image, run:

```bash
docker pull deepmi/fastsurfer
```

This downloads the newest official FastSurfer image for NVIDIA GPUs, with the default CUDA version of that release,
{{ CUDA_VERSION }}. To pull a specific version of FastSurfer or CUDA, specify the tag, for example:

```bash
docker pull deepmi/fastsurfer:{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}
```

In general, images are named and tagged as `deepmi/fastsurfer:<device>-v<version>`, where `<device>` is
`cu<cuda_version>` for NVIDIA GPUs with a specific CUDA version (e.g. `{{ CUDA_STRING }}`), `rocm<rocm_version>` for
AMD GPUs (experimental), or `cpu` without hardware acceleration (smaller and thus faster to download).
[Which image fits your GPU](../../doc/overview/install/LINUX.md#nvidia-gpus) explains the CUDA images, and
[Docker Hub](https://hub.docker.com/r/deepmi/fastsurfer/tags) lists the `<device>` options available for each version.
Similarly, `v<version>` is the version string, for example `v{{ FASTSURFER_VERSION }}`. `latest` points to the newest
NVIDIA image (with the default CUDA version of that release), `cpu-latest` to the newest `cpu` image, for example:

```bash
docker pull deepmi/fastsurfer:cpu-v{{ FASTSURFER_VERSION }}
```

Running the (official) Docker Image
-----------------------------------
After pulling the image, you can start a FastSurfer container and process a T1-weighted image (both segmentation and
surface reconstruction) with the following command:

```bash
freesurfer_license=${freesurfer_license:-/path/to/your/freesurfer/license_file}
docker run --gpus all -v $HOME/my_mri_data:$HOME/my_mri_data \
           -v $HOME/my_fastsurfer_analysis:$HOME/my_fastsurfer_analysis \
           -v $freesurfer_license:$freesurfer_license \
           --rm --user $(id -u):$(id -g) deepmi/fastsurfer:{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }} \
           --fs_license $freesurfer_license \
           --t1 $HOME/my_mri_data/subjectX/t1_weighted.nii.gz \
           --sid subjectX --sd $HOME/my_fastsurfer_analysis \
           --threads 4 --3T # and more flags
```

### Docker Flags
* `--gpus`: gives the container access to GPUs, and selects which ones. In the example above, `all` makes every GPU
  available to FastSurfer in the container. To use a single one (e.g. GPU 0), set `--gpus device=0`. To use several
  specific GPUs (e.g. GPU 0, 1 and 3), use `--gpus "device=0,1,3"`. Leave it out to run FastSurfer on the CPU.
* `-v`: defines which data is shared between the host system and the container, and how. By default, nothing is
  shared, so `-v` explicitly shares a directory or file. It follows the format `-v <host_dir>:<container_dir>:<options>`.
  In its simplest form, `<host_dir>` and `<container_dir>` are the same, so paths inside the container are the same as
  on the host. `:<options>` may be left out, or `:ro` makes the files read-only for the container. Share the input
  files, the output folder (subjects directory) and the FreeSurfer license.
* `--user $(id -u):$(id -g)`: the user and group the container runs as, which decides the access to files
  (**required**). `$(id -u)` and `$(id -g)` give your user and group IDs. Without this flag, FastSurfer exits with a
  message asking you to map your host user. Running the container as root (`--user 0:0`) is strongly discouraged and
  must be combined with the FastSurfer flag `--allow_root`.
* `--rm`: removes the container once the analysis has finished (optional, but recommended).
* `-d`: runs the container in detached mode, so you return to the shell without screen output (optional).

#### Advanced Docker Flags
* `--group-add <group_list>`: If additional user groups are required to access files, add them with
  `--group-add <group_id>[,...]` or `--group-add $(id -G <group_name>)`.

### FastSurfer Flags
In principle, these are the same as for [run_fastsurfer.sh](../../doc/scripts/RUN_FASTSURFER.md#required-arguments),
with the following changes:
* `--fs_license` cannot be detected automatically and must be passed. It is the path of your FreeSurfer license
  inside the container, so share the file with `-v` ([above](#docker-flags)).
* `--t1` and `--sd` are required, and also need to be shared with `-v` ([above](#docker-flags)).
* `--sid` is the subject ID (the name of the output folder), as for
  [run_fastsurfer.sh](../../doc/scripts/RUN_FASTSURFER.md).

A directory with the name specified in `--sid` (here subjectX) will be created in the output directory (specified via
`--sd`), so in this example, the output will be written to `$HOME/my_fastsurfer_analysis/subjectX/`. Make sure this
directory does not exist yet, to avoid overwriting existing files.

All other flags are the same as explained in the [run_fastsurfer.sh documentation](../../doc/scripts/RUN_FASTSURFER.md).

### Docker Best Practice
* Do not mount the user home directory into the Docker container as the home directory.

  Why? If the user inside the Docker container has access to a user directory, settings from that directory might
  bleed into the FastSurfer pipeline.

  How? Docker does not mount the home directory by default, so unless you manually set the `HOME` environment
  variable, all should be fine.

FastSurfer Docker Image Creation
--------------------------------
In `tools/Docker`, we provide a build script and a Dockerfile for users (usually developers) who want to build their
own Docker images, for these platforms:

* NVIDIA / CUDA (Example 1)
* CPU (Example 2)
* AMD / ROCm (experimental, Example 3)
* Intel / XPU (very experimental)

To run only the segmentation or only the surface reconstruction, pass `--seg_only` or `--surf_only` to FastSurfer.

For many HPC users with limited GPUs or with very large datasets, it may be most efficient to run the full pipeline on
the CPU, trading a longer runtime of the segmentation for massive parallelization on the subject level.

To run our Docker containers on an Intel Mac, increase the memory Docker Desktop may use, see
[Docker on Intel Macs](../../doc/overview/install/MACOS.md#docker-intel-macs) and
[Change Docker Desktop settings on Mac](https://docs.docker.com/desktop/settings/mac/). On a Mac with Apple silicon,
use the [macOS package](../../doc/overview/install/MACOS.md) instead: Docker cannot use the Apple GPU, the package uses
it automatically.

### General build settings
The build script `build.py` supports additional arguments, targets and options, see
`python tools/Docker/build.py --help`.

Besides selecting the build arguments, the build script creates the file `BUILD.info` in the FastSurfer root
directory, which FastSurfer uses to report its version (including the git hash of the source the image was built
from). The Docker build fails without this file.
With `--dry_run`, the build script prints the command instead of executing it, so you can also run
`python tools/Docker/build.py --device cuda --dry_run | bash`.

By default, the build script tags your image as `fastsurfer:<device>-v<version_tag>`, where `<version_tag>` is
`<version>_<git_hash>` (the version from pyproject.toml and the current git hash) and `<device>` is the value of
`--device` (`cuda` and `rocm` are replaced by their default versions, e.g. `{{ CUDA_STRING }}`). Specify a custom tag
with `--tag <image_tag>`.

By default, the Python environment is resolved from `pyproject.toml`, which allows the latest compatible dependency
versions. To build from the backend-neutral pinned `requirements.txt` instead, add `--pinned_requirements`. The
selected `--device` is still passed to `uv --torch-backend`, so the same pinned requirements file works for the CPU
and all supported CUDA versions, and the PyTorch wheels for the backend are selected during the build.

#### BuildKit
We recommend using BuildKit to build Docker images (e.g. `DOCKER_BUILDKIT=1`; `build.py` always adds this). To
install BuildKit, run
`wget -qO ~/.docker/cli-plugins/docker-buildx https://github.com/docker/buildx/releases/download/<buildx_version>/buildx-<buildx_version>.<platform>`,
for example
`wget -qO ~/.docker/cli-plugins/docker-buildx https://github.com/docker/buildx/releases/download/v0.12.1/buildx-v0.12.1.linux-amd64`.
See also https://github.com/docker/buildx#manual-download.

### Example 1: Build GPU FastSurfer Image
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

To run the analysis, use the same command as for the official image above, with your image:
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

### Example 2: Build CPU FastSurfer Image
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

### Example 3: Experimental Build for AMD GPUs
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
