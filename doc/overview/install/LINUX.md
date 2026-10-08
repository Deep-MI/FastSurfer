Linux
=====
On Linux, run FastSurfer in one of our container images: they contain the full pipeline (segmentation and surface
reconstruction) and all software it needs. The surface reconstruction also needs a
[FreeSurfer license](../INSTALL.md#freesurfer-license), which is free but not included: get it from FreeSurfer
before your first full run. We provide images for NVIDIA GPUs (CUDA), for AMD GPUs (ROCm, experimental)
and for the CPU only, on [Docker Hub](https://hub.docker.com/r/deepmi/fastsurfer).

A GPU with enough memory makes the segmentation much faster, see the
[system requirements](../intro.rst#system-requirements). Without a GPU, use the [CPU image](#cpu-only).

The images are tagged `deepmi/fastsurfer:<device>-v<version>`, where `<device>` is `cu<cuda_version>` for NVIDIA GPUs
(see [below](#nvidia-gpus)), `rocm<rocm_version>` for AMD GPUs (experimental), or `cpu` without GPU support (smaller
and thus faster to download), and `<version>` is the FastSurfer version. `latest` points to the newest NVIDIA image,
with the default CUDA version of that release, and `cpu-latest` to the newest CPU image.
[Docker Hub](https://hub.docker.com/r/deepmi/fastsurfer/tags) lists all tags. For reproducible results, use a
versioned tag.

NVIDIA GPUs
-----------
We build the images for two CUDA versions, because newer CUDA versions drop old GPUs and older ones lack the newest:

```{list-table}
:header-rows: 1
:widths: 21 10 51 18

* - Image
  - CUDA
  - GPUs
  - NVIDIA driver
* - `{{ CUDA_DEFAULT_STRING }}-v<version>` (default)
  - {{ CUDA_DEFAULT_VERSION }}
  - Turing (RTX 20, T4) to Blackwell (RTX 50, B200)
  - {{ CUDA_DRIVER }} or newer
* - `{{ CUDA_LEGACY_STRING }}-v<version>`
  - {{ CUDA_LEGACY_VERSION }}
  - Maxwell (GTX 900) to Hopper (H100), including Pascal (GTX 10, P100) and Volta (V100), but not Blackwell
  - {{ CUDA_LEGACY_DRIVER }} or newer
```

Use the default image unless your GPU is older than Turing or your driver is older than {{ CUDA_DRIVER }}; then use the
{{ CUDA_LEGACY_VERSION }} image. `nvidia-smi` shows the name of your GPU and the driver version.

If FastSurfer cannot use your GPU, because the image has no support for it, the driver is too old, or the container was
started without access to the GPU, it says so, names the image or flag to use instead, and runs on the CPU, which takes
much longer.

Apptainer or Singularity
------------------------
[Apptainer](https://apptainer.org) is the open-source continuation of Singularity, and both run the same images.
Apptainer also installs the `singularity` command, so the commands below work with either; with Apptainer, you can
also write `apptainer` instead of `singularity`.

With Apptainer (or Singularity) installed, build an image from our Docker image, for example in
`$HOME/my_singularity_images`:

```bash
mkdir -p $HOME/my_singularity_images
singularity build \
    $HOME/my_singularity_images/fastsurfer-{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}.sif \
    docker://deepmi/fastsurfer:{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}
```

[Running FastSurfer in a container](../CONTAINERS.md#using-apptainer-or-singularity) shows how to run FastSurfer with
it and explains the flags.

Docker
------
With Docker installed, download our image:

```bash
docker pull deepmi/fastsurfer:{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}
```

[Running FastSurfer in a container](../CONTAINERS.md#using-docker) shows how to run FastSurfer with it and explains
the flags.

For NVIDIA GPUs, Docker needs the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
In Docker's **rootless mode**, also follow its
[configuration for the rootless mode](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#rootless-mode),
otherwise Docker stops with
`docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`.

CPU only
--------
Without a GPU, use the CPU image, which is smaller. It runs the same pipeline, only the segmentation takes longer.
For Apptainer (or Singularity):

```bash
mkdir -p $HOME/my_singularity_images
singularity build \
    $HOME/my_singularity_images/fastsurfer-cpu-v{{ FASTSURFER_VERSION }}.sif \
    docker://deepmi/fastsurfer:cpu-v{{ FASTSURFER_VERSION }}
```

or, for Docker:

```bash
docker pull deepmi/fastsurfer:cpu-v{{ FASTSURFER_VERSION }}
```

Run it as described in [Running FastSurfer in a container](../CONTAINERS.md), without `--nv` (Apptainer or
Singularity) or `--gpus all` (Docker).

AMD GPUs (experimental)
-----------------------
We have successfully run the segmentation on an AMD GPU (Radeon Pro W6600) with ROCm. This needs a supported (or
semi-supported) GPU and the right kernel version. Install the AMD kernel modules on the host as described in the
[ROCm installation instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/), and add your user
to the groups they name.

Then download our image for ROCm {{ ROCM_VERSION }}:

```bash
docker pull deepmi/fastsurfer:{{ ROCM_STRING }}-v{{ FASTSURFER_VERSION }}
```

AMD needs a few more flags in the `docker run` command, see
[Running FastSurfer in a container](../CONTAINERS.md#using-docker) for `<docker_flags>` and `<fastsurfer_flags>`. To
build your own ROCm image, see [Building FastSurfer Docker images](../../developer/docker.rst).

Usage:

```text
docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        --device=/dev/kfd --device=/dev/dri --group-add video \
        --ipc=host --shm-size 8G \
        <docker_flags> deepmi/fastsurfer:{{ ROCM_STRING }}-v{{ FASTSURFER_VERSION }} \
                <fastsurfer_flags>
```

This image is experimental and uses different Python packages, so its results can differ from our validation
results. Check them visually.

Installing from source
----------------------
To install FastSurfer without containers, see [installing from source](NATIVE.md).
