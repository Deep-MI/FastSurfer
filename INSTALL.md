# Introduction

FastSurfer is a pipeline for the segmentation of human brain MRI data. It consists of two main components the FastSurferCNN or FastSurferVINN network for the fast segmentation of an MRI and the recon_surf script for the efficient creation of surfaces and basically all files and statistics that also FreeSurfer provides. 

The preferred way of installing and running FastSurfer is via Singularity or Docker containers. We provide pre-build images at Dockerhub for various application cases: i) for only the segmentation (both GPU and CPU), ii) for only the CPU-based recon-surf pipeline, and iii) for the full pipeline (GPU or CPU). 

We also provide information on a native installs on some operating systems, but since dependencies may vary, this can produce results different from our testing environment and we may not be able to support you if things don't work. 


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
docker pull deepmi/fastsurfer:gpu
```

Our [README](README.md) explains how to run FastSurfer (for the full pipeline you will also need a FreeSurfer .license file !) and you can find details on how to build your own Images here: [Docker](docker/README.md) and [Singularity](singularity/README.md). 


### Native - Ubuntu 20.04


#### System Packages

You will need a few additional packages that may yet be missing on your system (for this you need sudo access or ask a system admin):

```
sudo apt-get update && apt-get install -y --no-install-recommends \
      wget \
      git \
      ca-certificates \
      file
```

This is enough to run the FastSurfer neural network segmentation, if you want to run the full pipeline, you also need a working installation of FreeSurfer (and its dependencies).

#### FastSurfer

Get a FastSurfer version from GitHub. Here you can decide if you want to install the current experimental "dev" version (which can be broken) or the "stable" branch (that has been tested thoroughly).

#### Conda

The easiest way to install FastSurfer dependencies is with conda:

```
wget --no-check-certificate -qO ~/miniconda.sh https://repo.continuum.io/miniconda/$CONDA_FILE
chmod +x ~/miniconda.sh
sudo ~/miniconda.sh -b -p /opt/conda && \
rm ~/miniconda.sh 
```

Install dependencies into a new environment:

```
conda env create -f /fastsurfer/fastsurfer_env_gpu.yml 
```

(in the above step you can select from other fastsurfer...yml files for CPU and recon-surf-only versions).

Once all dependencies are installed, run the FastSurfer segmentation only (!!) by calling ```./run_fastsurfer.sh --seg_only ....``` with the appropriate command line flags, see the [README](README.md). 

To run the full pipeline, install also FreeSurfer v7.2 according to their [Instructions](https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads). There is a freesurfer email list, if you into problems during this step. 

## MacOS 

Currently only CPU based procesing is supported. GPU processing on Apple Silicon Chips is under developoment.

Recommended: Mac with Apple Silicon M Chip and 16 GB RAM
You can also run on older Intel chips but it will be 2-4 times slower. 


### Native

On modern Macs with the Apple Silicon M1 or M2 ARM-based chips, we recommend a native CPU install as it runs much faster than Docker in our tests. On Intel chips you can also use Docker (see below).

We exepct you to already have git and a recent bash (version > 4.0) installed, e.g. via the packet manager brew.
This installs brew and then bash:

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install bash
```

Clone FastSurfer:dev, create a python environment, activate it, upgrade pip and install the requirements:

```
git clone -b dev https://github.com/Deep-MI/FastSurfer.git
python3 -m venv $HOME/python-envs/fastsurfer 
source $HOME/python-envs/fastsurfer/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.mac.txt
```

If the last step fails, you may need to edit requirements.mac.txt and adjust version number to what is available. On newer
M1 Macs, we also had issues with the h5py pacakge, which could be solved by using brew for help:

```
brew install hdf5
export HDF5_DIR="$(brew --prefix hdf5)"
pip3 install --no-binary=h5py h5py
```

Once all dependencies are installed, run the FastSurfer segmentation only (!!) by calling ```bash ./run_fastsurfer.sh --seg_only ....``` with the appropriate command line flags, see the [README](README.md). 

To run the full pipeline, install also FreeSurfer v7.2 according to their [Instructions](https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads). There is a freesurfer email list, if you into problems during this step. 

### Docker (currently only Intel)

Docker can be used on Intel Macs as it should be similarly fast than a native install there.

First install Docker Desktop for Mac from https://docs.docker.com/get-docker/
Start it and under Preferences -> Resources set Memory to 15 GB (or the largest you have, if you are below 15GB, it may fail). Pull one of our pre-compiled Docker containers. For that open a terminal window and copy this command:

```
docker pull deepmi/fastsurfer
```

and run is as the example in our [README](README.md). 


## Windows

Nothing has been tested so far on Windows. We expect the CPU-based containers to work here. GPU passthrough will be explored in the future. If you want to make use of your GPU, you need to install a dual-boot with Ubuntu on your system.

