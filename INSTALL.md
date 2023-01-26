# Introduction

FastSurfer is a pipeline for the segmentation of human brain MRI data. It consists of two main components the FastSurferCNN or FastSurferVINN network for the fast segmentation of an MRI and the recon_surf script for the efficient creation of surfaces and basically all files and statistics that also FreeSurfer provides. 

The preferred way of installing and running FastSurfer is via Singularity or Docker containers. We provide pre-build images at Dockerhub for various application cases: i) for only the segmentation (both GPU and CPU), ii) for only the CPU-based recon-surf pipeline, and iii) for the full pipeline (GPU or CPU). 

We also provide information on a native install on some operating systems, but since dependencies may vary, this can produce results different from our testing environment and we may not be able to support you if things don't work. Our testing is performed on Ubuntu 20.04 via our provided Docker images.


# Installation

## Linux

Recommendation: 8 GB CPU RAM, NVIDIA GPU with 8GB RAM
(minimal: 2GB GPU RAM can work with special flags).

You can also run a non-GPU (CPU only) but that will be much slower. 
Non NVIDIA GPU architectures are not supported. 

### Singularity

Assuming you have singularity installed already (by a system admin), you can build an image easily from our Dockerhub images. Run this command from a directory where you want to store singularity images:

```
singularity build fastsurfer-gpu.sif docker://deepmi/fastsurfer
```

Our [README](README.md) explains how to run FastSurfer (for the full pipeline you will also need a FreeSurfer .license file !) and you can find details on how to build your own Images here: [Docker](docker/README.md) and [Singularity](singularity/README.md). 


### Docker

This is very similar to Singularity. Assuming you have Docker installed (by a system admin) you just need to pull one of our pre-build Docker images from dockerhub:

```
docker pull deepmi/fastsurfer:gpu-v2.0.0
```

Our [README](README.md) explains how to run FastSurfer (for the full pipeline you will also need a FreeSurfer .license file !) and you can find details on how to build your own Images here: [Docker](docker/README.md) and [Singularity](singularity/README.md). 


### Native - Ubuntu 20.04

In a native install you need to install all dependencies (distro packages, FreeSurfer in the supported version, python dependencies) yourself. Here we will walk you through what you need.

#### System Packages

You will need a few additional packages that may yet be missing on your system (for this you need sudo access or ask a system admin):

```
sudo apt-get update && apt-get install -y --no-install-recommends \
      wget \
      git \
      ca-certificates \
      file
```

You also need a working version of python3 (currently we test with version 3.8). These packages shoule be sufficient to install python dependencies and then run the FastSurfer neural network segmentation. If you want to run the full pipeline, you also need a working installation of FreeSurfer (and its dependencies).

#### FastSurfer

Get a FastSurfer version from GitHub. Here you can decide if you want to install the current experimental "dev" version (which can be broken) or the "stable" branch (that has been tested thoroughly).

```
git clone https://github.com/Deep-MI/FastSurfer.git
cd FastSurfer
```

Get a FastSurfer version from GitHub. Here you can decide if you want to install the current experimental "dev" version (which can be broken) or the "stable" branch (that has been tested thoroughly).

#### Conda

The easiest way to install FastSurfer dependencies is with conda. If you don't have conda on your system, an admin needs to install it:

```
wget --no-check-certificate -qO ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh
chmod +x ~/miniconda.sh
sudo ~/miniconda.sh -b -p /opt/conda && \
rm ~/miniconda.sh 
```

Install dependencies into a new environment:

```
conda env create -f ./fastsurfer_env_gpu.yml 
conda activate fastsurfer_gpu
```

(in the above step you can select from other fastsurfer...yml files for CPU and recon-surf-only versions).

You should also make sure that all network checkpoint files are downloaded at install time:
```
python3 ./FastSurferCNN/download_checkpoints.py --all
```

Once all dependencies are installed, run the FastSurfer segmentation only (!!) by calling ```./run_fastsurfer.sh --seg_only ....``` with the appropriate command line flags, see the [README](README.md). 

To run the full pipeline, install also the supported FreeSurfer version according to their [Instructions](https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads). There is a freesurfer email list, if you run into problems during this step. 

### AMD GPUs (experimental)

We have successfully run the segmentation on an AMD GPU (Radeon Pro W6600) using ROCm. For this to work you need to make sure you are using a supported (or semi-supported) GPU and the correct kernel version. AMD kernel modules need to be installed on the host system according to ROCm installation instructions and additional groups need to be setup and user added. 
See https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.2.3/page/Introduction_to_AMD_ROCm_Installation_Guide_for_Linux.html 
Then you can build a Docker with ROCm support and run it.

```
docker build --rm=true -t deepmi/fastsurfer:amd -f ./Docker/Dockerfile_FastSurferCNN_AMD .
docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd \
           --device=/dev/dri --group-add video --ipc=host --shm-size 8G \
                      -v /home/user/my_mri_data:/data \
                      -v /home/user/my_fastsurfer_analysis:/output \
                      -v /home/user/my_fs_license_dir:/fs_license \
                      --rm --user $(id -u):$(id -g) deepmi/fastsurfer:amd \
                      --fs_license /fs_license/license.txt \
                      --t1 /data/subjectX/orig.mgz \
                      --sid subjectX --sd /output \
                      --seg_only
```
Note, that this is using an older Python version and packages, so results can differ from our validation results. So please do visual QC.

## MacOS 

Processing on Mac CPUs (both Intel and Apple Silicon) is possible. On Apple Silicon you can even use the GPU (experimental) by passing ```--device mps```

Recommended: Mac with Apple Silicon M Chip and 16 GB RAM
You can also run on older Intel CPUs but it will be 2-4 times slower. 

### Native

On modern Macs with the Apple Silicon M1 or M2 ARM-based chips, we recommend a native install as it runs much faster than Docker in our tests. It is also the only way to make use of the built-in GPU. On Intel chips you can use either native install or Docker (see below).

We expect you to already have git and a recent bash (version > 4.0 required!) installed, e.g. via the packet manager brew.
This installs brew and then bash:

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install bash
```

Make sure you use this bash and not the older one provided with MacOS!
Clone FastSurfer, create a python environment, activate it, upgrade pip and install the requirements. Here we use pip, but you should also be able to use conda as described above: 

```
git clone https://github.com/Deep-MI/FastSurfer.git
python3 -m venv $HOME/python-envs/fastsurfer 
source $HOME/python-envs/fastsurfer/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.mac.txt
```

If the last step fails, you may need to edit ```requirements.mac.txt``` and adjust version number to what is available. On newer
M1 Macs, we also had issues with the h5py pacakge, which could be solved by using brew for help (not sure this is needed any longer):

```
brew install hdf5
export HDF5_DIR="$(brew --prefix hdf5)"
pip3 install --no-binary=h5py h5py
```

You should also make sure that all network checkpoint files are downloaded at install time:
```
python3 FastSurferCNN/download_checkpoints.py --all
```

Once all dependencies are installed, run the FastSurfer segmentation only (!!) by calling ```bash ./run_fastsurfer.sh --seg_only ....``` with the appropriate command line flags, see the [README](README.md). 

You can also try out running on the Apple Silicon GPU by:

```
export PYTORCH_ENABLE_MPS_FALLBACK=1
./run_fastsurfer.sh --seg_only --device mps ....
```

This will be at least twice as fast as CPU. The fallback environment variable is necessary as one function is not yet implemented for 
the GPU and will fall back to CPU. Note: you may need to prepend the command with ```bash ./run_fastsurfer.sh ...``` and ensure that the installed bash is used instead of the system one. 

To run the full pipeline, install and source also the supported FreeSurfer version according to their [Instructions](https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads). There is a freesurfer email list, if you run into problems during this step. 

### Docker (currently only Intel)

Docker can be used on Intel Macs as it should be similarly fast as a native install there. It would allow you to run the full pipeline.

First install Docker Desktop for Mac from https://docs.docker.com/get-docker/
Start it and under Preferences -> Resources set Memory to 15 GB (or the largest you have, if you are below 15GB, it may fail). Pull one of our pre-compiled Docker containers. For that open a terminal window and copy this command:

```
docker pull deepmi/fastsurfer
```

and run is as the example in our [README](./README.md). 


## Windows

### Docker (CPU version)

In order to run Fastsurfer on your Windows system using docker make sure that:
* you have [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
* as well as [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/) installed and running

After everything is installed, start Windows PowerShell and run the following command to pull the CPU Docker image:

```bash
docker pull deepmi/fastsurfer:cpu-v2.0.0
```

Now you can run Fastsurfer the same way as described in our [Docker README](./Docker/README.md) for the CPU build
```bash
docker run -v C:/Users/user/my_mri_data:/data \
           -v C:/Users/user/my_fastsurfer_analysis:/output \
           -v C:/Users/user/my_fs_license_dir:/fs_license \
           --rm --user $(id -u):$(id -g) deepmi/fastsurfer:cpu-v2.0.0 \
           --fs_license /fs_license/license.txt \
           --t1 /data/subjectX/orig.mgz \
           --device cpu \
           --sid subjectX --sd /output \
           --parallel
```
Note the [system requirements](https://github.com/Deep-MI/FastSurfer#system-requirements) of at least 8GB of RAM for the CPU version. If the process fails, check if your [WSL2 distribution has enough memory reserved](https://www.aleksandrhovhannisyan.com/blog/limiting-memory-usage-in-wsl-2/).

This was tested using Windows 10 Pro version 21H1 and the WSL Ubuntu 20.04  distribution

### Docker (GPU version)

In addition to the requirements from the CPU version, you also need to make sure that:
* you have Windows 11 or Windows 10 21H2 or greater,
* the latest WSL Kernel or at least 4.19.121+ (5.10.16.3 or later for better performance and functional fixes),
* an NVIDIA GPU and the latest [NVIDIA CUDA driver](https://developer.nvidia.com/cuda/wsl)
* CUDA toolkit installed on WSL. See: _[CUDA Support for WSL 2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2)_

Follow the following tutorial for installing the correct drivers and software: [Enable NVIDIA CUDA on WSL](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)

After everything is installed, start Windows PowerShell and run the following command to pull the GPU Docker image:

```bash
docker pull deepmi/fastsurfer:latest
```

Now you can run Fastsurfer the same way as described in our [Docker README](./Docker/README.md)
```bash
docker run --gpus all
           -v C:/Users/user/my_mri_data:/data \
           -v C:/Users/user/my_fastsurfer_analysis:/output \
           -v C:/Users/user/my_fs_license_dir:/fs_license \
           --rm --user $(id -u):$(id -g) deepmi/fastsurfer:latest \
           --fs_license /fs_license/license.txt \
           --t1 /data/subjectX/orig.mgz \
           --sid subjectX --sd /output \
           --parallel
```

Note the [system requirements](https://github.com/Deep-MI/FastSurfer#system-requirements) of at least 7 GB CPU RAM and 2 GB GPU RAM for the GPU version. If the process fails, check if your [WSL2 distribution has enough memory reserved](https://www.aleksandrhovhannisyan.com/blog/limiting-memory-usage-in-wsl-2/).

