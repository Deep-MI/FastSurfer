Installation
============
FastSurfer is a pipeline for the segmentation of human brain MRI data. It consists of two main components: the networks for the fast segmentation of an MRI (FastSurferVINN, CerebNet, ...) and the recon_surf script for the efficient creation of surfaces and most files and statistics that also FreeSurfer provides.

The preferred way of installing and running FastSurfer is via Singularity or Docker containers on a Linux host system (with a GPU). We provide pre-build images at Dockerhub for various application cases: i) for only the segmentation (both GPU and CPU), ii) for only the CPU-based recon-surf pipeline, and iii) for the full pipeline (GPU or CPU).

We also provide information on a native install on some operating systems, but since dependencies may vary, this can produce results different from our testing environment and we may not be able to support you if things don't work. Our testing is performed on Ubuntu 22.04 via our provided Docker images.


Linux
-----
Recommended System Spec: 8 GB system memory, NVIDIA GPU with 8 GB graphics memory.

Minimum System Spec: 8 GB system memory (this requires running FastSurfer on the CPU only, which is much slower)

Non-NVIDIA GPU architectures (AMD) are experimental and not officially supported, but seem to work well also.

### Singularity (or Apptainer)
Assuming you have singularity installed already (by a system admin), you can build a Singularity image easily from our Dockerhub images. Run this command from a directory where you want to store singularity images:

```bash
singularity build fastsurfer-gpu.sif docker://deepmi/fastsurfer:latest
```
Additionally, [the Singularity documentation](SINGULARITY.md) contains detailed directions for building your own Singularity images from Docker.

[Example 1](EXAMPLES.md#example-1-fastsurfer-singularity-or-apptainer) explains how to run FastSurfer (for the full pipeline you will also need a FreeSurfer .license file!) and you can find details on how to build your own images here: [Docker](../../tools/Docker/README.md) and [Singularity](SINGULARITY.md).


### Docker
This is very similar to Singularity. Assuming you have Docker installed (by a system admin) you just need to pull one of our pre-build Docker images from dockerhub:

```bash
docker pull deepmi/fastsurfer:latest
```

[Example 2](EXAMPLES.md#example-2-fastsurfer-docker) explains how to run FastSurfer (for the full pipeline you will also need a FreeSurfer .license file!) and you can find details on how to [build your own image](../../tools/Docker/README.md).

If you are using the **rootless mode**, you have to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and follow the [configuration for the rootless mode](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#rootless-mode). Otherwise, running FastSurfer with Docker will give you this error message ```docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]```.


### Native (Ubuntu 20.04 or Ubuntu 22.04)
In a native install you need to install all dependencies (distro packages, FreeSurfer in the supported version, python dependencies) yourself. Here we will walk you through what you need.

#### 1. System Packages
You will need a few additional packages that may be missing on your system (for this you need sudo access or ask a system admin):

```bash
sudo apt-get update && apt-get install -y --no-install-recommends \
      wget \
      git \
      ca-certificates \
      file
```

If you are using **Ubuntu 20.04**, you will need to upgrade to a newer version of libstdc++, as some 'newer' python packages need GLIBCXX 3.4.29, which is not distributed with Ubuntu 20.04 by default.

```bash
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt install -y g++-11
```

You also need to have bash-3.2 or higher (check with `bash --version`).

You also need a working version of python3 (we do not support other versions). These packages should be sufficient to install python dependencies and then run the FastSurfer neural network segmentation. If you want to run the full pipeline, you also need a [working installation of FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads) (including its dependencies and a license file).

If you are using pip, make sure pip is updated as older versions will fail.

#### 2. uv for python

We recommend to install uv as your python environment and package manager. [uv](https://docs.astral.sh/uv/) is a very
fast package manager, which makes managing different environments even easier. See
[uv's documentation](https://docs.astral.sh/uv/getting-started/installation/) for more information on installation such
as [autocompletion info](https://docs.astral.sh/uv/getting-started/installation/#shell-autocompletion).

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

#### 3. FastSurfer
Get FastSurfer from GitHub. Here you can decide if you want to install the current experimental "dev" version (which can be broken) or the "stable" branch (that has been tested thoroughly):

```bash
cd /path/to/install
# FastSurfer will get cloned to /path/to/install/FastSurfer
git clone --branch stable https://github.com/Deep-MI/FastSurfer.git
cd FastSurfer
```

#### 4. Python environment
Create a new environment and install FastSurfer dependencies:

```bash
# make sure you are in the FastSurfer directory!
# create a .venv environment directory inside /path/to/install/FastSurfer with the FastSurfer dependencies
# the minimum required python version is 3.10
uv venv --python python3.12
# download and install packages for the fastsurfer environment (implicitly read from requirements.txt)
uv pip sync pyproject.toml
```
`uv` will also try to find the correct backend for your hardware, but you can manually specify the backend for testing:
purposes:
```bash
# make sure you are in the FastSurfer directory!
uv pip sync pyproject.toml --torch-backend cpu
```
You can now activate the FastSurfer environment with
```bash
source .venv/bin/activate
```

Next, add the fastsurfer directory to the python path:
```bash
# make sure you are in the FastSurfer directory!
export PYTHONPATH="${PYTHONPATH}:$PWD"
```

This will need to be done every time you want to run FastSurfer, or you need to add this line to your `~/.bashrc` if you are using bash, for example:
```bash
# make sure you are in the FastSurfer directory!
echo "export PYTHONPATH=\"\${PYTHONPATH}:$(pwd)\"" >> ~/.bashrc
```

You can also download all network checkpoint files (this should be done if you are installing for multiple users):
```bash
# make sure you are in the FastSurfer directory!
python FastSurferCNN/download_checkpoints.py --all
```

Once all dependencies are installed, you are ready to run the FastSurfer segmentation-only (!!) pipeline by calling ```./run_fastsurfer.sh --seg_only ....``` , see [Example 3](EXAMPLES.md#example-3-native-fastsurfer-on-subjectx-with-parallel-processing-of-hemis) for command line flags.

#### 5. FreeSurfer
To run the full pipeline, you will need to install FreeSurfer (we recommend and support version 7.4.1) according to their [Instructions](https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads). There is a freesurfer email list, if you run into problems during this step.

Make sure, the `${FREESURFER_HOME}` environment variable is set, so FastSurfer finds the FreeSurfer binaries.

### AMD GPUs (experimental)
We have successfully run the segmentation on an AMD GPU (Radeon Pro W6600) using ROCm. For this to work you need to make sure you are using a supported (or semi-supported) GPU and the correct kernel version. AMD kernel modules need to be installed on the host system according to ROCm installation instructions and additional groups need to be setup and your user added to them, see https://rocm.docs.amd.com/projects/install-on-linux/en/latest/ .

Build the Docker container with ROCm support.

```bash
python tools/Docker/build.py --device rocm --tag my_fastsurfer:rocm
```

You will need to add a couple of flags to your docker run command for AMD, see [Example 2](EXAMPLES.md#example-2-fastsurfer-docker) for `**other-docker-flags**` or `*<*fastsurfer-flags*>*`:
```bash
docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd \
        --device=/dev/dri --group-add video --ipc=host --shm-size 8G \
        **other-docker-flags** my_fastsurfer:rocm \
                **fastsurfer-flags**
```
Note, that this docker image is experimental, uses a different Python version and python packages, so results can differ from our validation results. Please do visual QC.

MacOS
-----
Processing on Mac CPUs is possible. On Apple Silicon, you can even use the GPU by passing ```--device mps```.

Recommended System Spec: Mac with Apple Silicon M-Chip and 16 GB system memory.

For older Intel CPUs, we only support cpu-only, which will be 2-4 times slower.

### Docker (currently only supported for Intel CPUs)
Docker can be used on Intel Macs as it should be similarly fast as a native install there. It would allow you to run the full pipeline.

First, install [Docker Desktop for Mac](https://docs.docker.com/get-docker/).
Start it and set Memory to 15 GB under Preferences -> Resources (or the largest you have, if you are below 15GB, it may fail).

Second, pull one of our Docker containers. Open a terminal window and run:

```sh
docker pull deepmi/fastsurfer:latest
```

Continue with the example in [Example 2](EXAMPLES.md#example-2-fastsurfer-docker).

### Package

#### 1. Requirements
FastSurfer requires pre-installed python3.10+ and bash (at least 3.2).
You can install these via the packet manager brew.
To install brew and then python3.10, execute the following in a Terminal:

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.10
```

#### 2. FastSurfer package
From version 2.5 onward, FastSurfer ships a macOS installer package, which you can download from
[github](https://github.com/Deep-MI/FastSurfer/releases/).
There are package installers for both the Apple M-chip architecture (`arm64`) and for legacy Intel chips (`x86_64`).
To install, double-click the installer and follow the installer instructions.

After installation, you can find the FastSurfer applet, its source code, and selected FreeSurfer executables in the `/Applications` folder.

#### 3. Launching FastSurfer

To launch a configured FastSurfer terminal session, start the FastSurfer applet from Applications.

Once the terminal opens, it is already configured and you can easily run the full FastSurfer pipeline by typing and executing `run_fastsurfer.sh <fastsurfer-flags>`, where you replace `<fastsurfer-flags>` with the appropriate [commandline flags of FastSurfer](../../README.md#usage).

#### 4. Apple AI Accelerator support
On modern M-Chips you can try the Apple Silicon AI Accelerator by setting `PYTORCH_ENABLE_MPS_FALLBACK` and passing `--device mps` for the segmentation module to make use of the fast GPU:

```sh
export PYTORCH_ENABLE_MPS_FALLBACK=1
./run_fastsurfer.sh --seg_only --device mps ....
```

This will be at least twice as fast as `--device cpu`. Currently setting the fallback environment variable is necessary as `aten::max_unpool2d` is not yet implemented for MPS and will fall back to CPU.

Windows
-------

### Docker (CPU version)
In order to run FastSurfer on your Windows system using docker make sure that you have:
* [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
* [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/)

installed and running.

After everything is installed, start Windows PowerShell and run the following command to pull the CPU Docker image (check on [dockerhub](https://hub.docker.com/r/deepmi/fastsurfer/tags) what version tag is most recent for cpu):

```bash
docker pull deepmi/fastsurfer:cpu-latest
```

Now you can run Fastsurfer the same way as described in [Example 2](EXAMPLES.md#example-2-fastsurfer-docker) for the CPU build, for example:
```bash
docker run -v C:/Users/user/my_mri_data:/data \
           -v C:/Users/user/my_fastsurfer_analysis:/output \
           -v C:/Users/user/my_fs_license_dir:/fs_license \
           --rm --user $(id -u):$(id -g) deepmi/fastsurfer:cpu-latest \
           --fs_license /fs_license/license.txt \
           --t1 /data/subjectX/orig.mgz \
           --device cpu \
           --sid subjectX --sd /output
```
Note, the [system requirements](https://github.com/Deep-MI/FastSurfer#system-requirements) of at least 8GB of RAM for the CPU version. If the process fails, check if your [WSL2 distribution has enough memory reserved](https://www.aleksandrhovhannisyan.com/blog/limiting-memory-usage-in-wsl-2/).

This was tested using Windows 10 Pro version 21H1 and the WSL Ubuntu 20.04  distribution

### Docker (GPU version)
In addition to the requirements from the CPU version, you also need to make sure that you have:
* Windows 11 or Windows 10 21H2 or greater,
* the latest WSL Kernel or at least 4.19.121+ (5.10.16.3 or later for better performance and functional fixes),
* an NVIDIA GPU and the latest [NVIDIA CUDA driver](https://developer.nvidia.com/cuda/wsl)
* CUDA toolkit installed on WSL, see: _[CUDA Support for WSL 2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2)_

Follow [Enable NVIDIA CUDA on WSL](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl) to install the correct drivers and software.

After everything is installed, start Windows PowerShell and run the following command to pull the GPU Docker image:

```bash
docker pull deepmi/fastsurfer:latest
```

Now you can run Fastsurfer the same way as described in [Example 2](EXAMPLES.md#example-2-fastsurfer-docker), for example:
```bash
docker run --gpus all \
           -v C:/Users/user/my_mri_data:/data \
           -v C:/Users/user/my_fastsurfer_analysis:/output \
           -v C:/Users/user/my_fs_license_dir:/fs_license \
           --rm --user $(id -u):$(id -g) deepmi/fastsurfer:latest \
           --fs_license /fs_license/license.txt \
           --t1 /data/subjectX/orig.mgz \
           --sid subjectX --sd /output
```

Note the [system requirements](https://github.com/Deep-MI/FastSurfer#system-requirements) of at least 8 GB system memory and 2 GB graphics memory for the GPU version. If the process fails, check if your [WSL2 distribution has enough memory reserved](https://www.aleksandrhovhannisyan.com/blog/limiting-memory-usage-in-wsl-2/).
