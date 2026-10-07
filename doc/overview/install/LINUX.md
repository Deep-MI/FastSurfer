Linux
=====
On Linux, run FastSurfer in one of our container images: they contain the full pipeline (segmentation and surface
reconstruction) and everything it needs. We provide images for NVIDIA GPUs (CUDA), for AMD GPUs (ROCm, experimental)
and for the CPU only, on [Docker Hub](https://hub.docker.com/r/deepmi/fastsurfer). Use Docker on your own
workstation and Singularity (or Apptainer) on a compute cluster, where Docker is usually not available.

A GPU with enough memory makes the segmentation much faster, see the
[system requirements](../intro.rst#system-requirements). Without a GPU, use the [CPU image](#cpu-only).

Docker
------
With Docker installed (by you or a system admin), download our image:

```bash
docker pull deepmi/fastsurfer:{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}
```

[Example 2](../EXAMPLES.md#example-2-fastsurfer-docker) shows how to run FastSurfer with it (the full pipeline also
needs a [FreeSurfer license](../INSTALL.md#freesurfer-license)). The [Docker page](../docker.rst) explains the
Docker flags and how to build your own image.

For NVIDIA GPUs, Docker needs the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
In Docker's **rootless mode**, also follow its
[configuration for the rootless mode](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#rootless-mode),
otherwise Docker stops with
`docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`.

Singularity or Apptainer
------------------------
With Singularity (or [Apptainer](https://apptainer.org)) installed, usually by a system admin, build a Singularity
image from our Docker image, for example in `$HOME/my_singularity_images`:

```bash
mkdir -p $HOME/my_singularity_images
singularity build \
    $HOME/my_singularity_images/fastsurfer-{{ FASTSURFER_VERSION }}.sif \
    docker://deepmi/fastsurfer:{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}
```

[Example 1](../EXAMPLES.md#example-1-fastsurfer-singularity-or-apptainer) shows how to run FastSurfer with it (the
full pipeline also needs a [FreeSurfer license](../INSTALL.md#freesurfer-license)), and the
[Singularity page](../SINGULARITY.md) explains the Singularity flags and how to build your own image.

CPU only
--------
Without a GPU, use the CPU image, which is smaller. It runs the same pipeline, only the segmentation takes longer.

```bash
docker pull deepmi/fastsurfer:cpu-v{{ FASTSURFER_VERSION }}
```

or, for Singularity:

```bash
mkdir -p $HOME/my_singularity_images
singularity build \
    $HOME/my_singularity_images/fastsurfer-cpu-{{ FASTSURFER_VERSION }}.sif \
    docker://deepmi/fastsurfer:cpu-v{{ FASTSURFER_VERSION }}
```

Run it as in the examples, without `--gpus all` (Docker) or `--nv` (Singularity).

AMD GPUs (experimental)
-----------------------
We have successfully run the segmentation on an AMD GPU (Radeon Pro W6600) with ROCm. This needs a supported (or
semi-supported) GPU and the right kernel version. Install the AMD kernel modules on the host as described in the
[ROCm installation instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/), and add your user
to the groups they name.

Build the Docker image with ROCm support, for example:

```bash
export FASTSURFER_HOME=${FASTSURFER_HOME:-/path/to/FastSurfer}
python3 $FASTSURFER_HOME/tools/Docker/build.py --device rocm \
    --tag my_fastsurfer:rocm
```

AMD needs a few more flags in the `docker run` command, see
[Example 2](../EXAMPLES.md#example-2-fastsurfer-docker) for `<docker_flags>` and `<fastsurfer_flags>`.

Usage:

```text
docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        --device=/dev/kfd --device=/dev/dri --group-add video \
        --ipc=host --shm-size 8G \
        <docker_flags> my_fastsurfer:rocm \
                <fastsurfer_flags>
```

This image is experimental and uses different Python packages, so its results can differ from our validation
results. Check them visually.

Installing from source
----------------------
To install FastSurfer without containers, see [installing from source](NATIVE.md).
