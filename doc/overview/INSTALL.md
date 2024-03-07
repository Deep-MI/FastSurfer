# Introduction

FastSurfer is a pipeline for the segmentation of human brain MRI data. It consists of two main components: the networks for the fast segmentation of an MRI (FastSurferVINN, CerebNet) and the recon_surf script for the efficient creation of surfaces and most files and statistics that also FreeSurfer provides. 

The preferred way of installing and running FastSurfer is via Singularity or Docker containers. We provide pre-build images at Dockerhub for various application cases: i) for only the segmentation (both GPU and CPU), ii) for only the CPU-based recon-surf pipeline, and iii) for the full pipeline (GPU or CPU). 

We also provide information on a native install on some operating systems, but since dependencies may vary, this can produce results different from our testing environment and we may not be able to support you if things don't work. Our testing is performed on Ubuntu 20.04 via our provided Docker images.


# Installation

## Linux

Recommended System Spec: 8 GB system memory, NVIDIA GPU with 8 GB graphics memory.

Minimum System Spec: 8 GB system memory (this requires running FastSurfer on the CPU only, which is much slower) 

Non-NVIDIA GPU architectures (Apple M1, AMD) are not officially supported, but experimental. 

### Singularity

Assuming you have singularity installed already (by a system admin), you can build an image easily from our Dockerhub images. Run this command from a directory where you want to store singularity images:

```bash
singularity build fastsurfer-gpu.sif docker://deepmi/fastsurfer:latest
```
Additionally, [the Singularity README](Singularity/README.md) contains detailed directions for building your own Singularity images from Docker.

Our [README](README.md#example-2--fastsurfer-singularity) explains how to run FastSurfer (for the full pipeline you will also need a FreeSurfer .license file!) and you can find details on how to build your own images here: [Docker](Docker/README.md) and [Singularity](Singularity/README.md). 


### Docker

This is very similar to Singularity. Assuming you have Docker installed (by a system admin) you just need to pull one of our pre-build Docker images from dockerhub:

```bash
docker pull deepmi/fastsurfer:latest
```

Our [README](README.md#example-1--fastsurfer-docker) explains how to run FastSurfer (for the full pipeline you will also need a FreeSurfer .license file!) and you can find details on how to [build your own image](Docker/README.md). 


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

You also need to have bash-4.0 or higher (check with `bash --version`). 

You also need a working version of python3 (we recommend python 3.10 -- we do not support other versions). These packages should be sufficient to install python dependencies and then run the FastSurfer neural network segmentation. If you want to run the full pipeline, you also need a [working installation of FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads) (including its dependencies).

If you are using pip, make sure pip is updated as older versions will fail.

#### 2. Conda for python

We recommend to install conda as your python environment. If you don't have conda on your system, an admin needs to install it:

```bash
wget --no-check-certificate -qO ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh
chmod +x ~/miniconda.sh
sudo ~/miniconda.sh -b -p /opt/conda && \
rm ~/miniconda.sh 
```

#### 3. FastSurfer
Get FastSurfer from GitHub. Here you can decide if you want to install the current experimental "dev" version (which can be broken) or the "stable" branch (that has been tested thoroughly):

```bash
git clone --branch stable https://github.com/Deep-MI/FastSurfer.git
cd FastSurfer
```

#### 4. Python environment

Create a new environment and install FastSurfer dependencies:

```bash
conda env create -f ./env/fastsurfer.yml 
conda activate fastsurfer
```

If you do not have an NVIDIA GPU, you can create appropriate ymls on the fly with `python ./Docker/install_env.py -m $MODE -i ./env/FastSurfer.yml -o ./fastsurfer_$MODE.yml`. Here `$MODE` can be for example `cpu`, see also `python ./Docker/install_env.py --help` for other options like rocm or cuda versions. Finally, replace `./env/fastsurfer.yml`  with your custom environment file `./fastsurfer_$MODE.yml`.
If you only want to run the surface pipeline, use `./env/fastsurfer_reconsurf.yml`.

Next, add the fastsurfer directory to the python path (make sure you have changed into it already):
```bash
export PYTHONPATH="${PYTHONPATH}:$PWD"
```

This will need to be done every time you want to run FastSurfer, or you need to add this line to your `~/.bashrc` if you are using bash, for example:
```bash
echo "export PYTHONPATH=\"\${PYTHONPATH}:$PWD\"" >> ~/.bashrc
```

You can also download all network checkpoint files (this should be done if you are installing for multiple users):
```bash
python3 FastSurferCNN/download_checkpoints.py --all
```

Once all dependencies are installed, you are ready to run the FastSurfer segmentation-only (!!) pipeline by calling ```./run_fastsurfer.sh --seg_only ....``` , see the [README](README.md#example-3--native-fastsurfer-on-subjectx--with-parallel-processing-of-hemis-) for command line flags.

#### 5. FreeSurfer
To run the full pipeline, you will need to install FreeSurfer (we recommend and support version 7.3.2) according to their [Instructions](https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads). There is a freesurfer email list, if you run into problems during this step. 

Make sure, the `${FREESURFER_HOME}` environment variable is set, so FastSurfer finds the FreeSurfer binaries.

### AMD GPUs (experimental)

We have successfully run the segmentation on an AMD GPU (Radeon Pro W6600) using ROCm. For this to work you need to make sure you are using a supported (or semi-supported) GPU and the correct kernel version. AMD kernel modules need to be installed on the host system according to ROCm installation instructions and additional groups need to be setup and your user added to them, see https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.2.3/page/Introduction_to_AMD_ROCm_Installation_Guide_for_Linux.html .

Build the Docker container with ROCm support.

```bash
python Docker/build.py --device rocm --tag my_fastsurfer:rocm
```

You will need to add a couple of flags to your docker run command for AMD, see [the Readme](README.md#example-1--fastsurfer-docker) for `**other-docker-flags**` or `**fastsurfer-flags**`:
```bash
docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd \
        --device=/dev/dri --group-add video --ipc=host --shm-size 8G \
        **other-docker-flags** my_fastsurfer:rocm \
                **fastsurfer-flags**
```
Note, that this docker image is experimental, uses a different Python version and python packages, so results can differ from our validation results. Please do visual QC.

## MacOS 

Processing on Mac CPUs is possible. On Apple Silicon, you can even use the GPU (experimental) by passing ```--device mps```.

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

Continue with the example in our [README](README.md#example-1--fastsurfer-docker). 


### Native

On modern Macs with the Apple Silicon M1 or M2 ARM-based chips, we recommend a native installation as it runs much faster than Docker in our tests. The experimental support for the built-in AI Accelerator is also only available on native installations. Native installation also supports older Intel chips.

#### 1. Git and Bash
If you do not have git and a recent bash (version > 4.0 required!) installed, install them via the packet manager, e.g. brew.
This installs brew and then bash:

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install bash
```

Make sure you use this bash and not the older one provided with MacOS!

#### 2. Python
Create a python environment, activate it, and upgrade pip. Here we use pip, but you should also be able to use conda for python: 

```sh
python3 -m venv $HOME/python-envs/fastsurfer 
source $HOME/python-envs/fastsurfer/bin/activate
python3 -m pip install --upgrade pip
```

#### 3. FastSurfer and Requirements
Clone FastSurfer:
```sh
git clone --branch stable https://github.com/Deep-MI/FastSurfer.git
cd FastSurfer
export PYTHONPATH="${PYTHONPATH}:$PWD"
```


Install the FastSurfer requirements
```sh
python3 -m pip install -r requirements.mac.txt
```

If this step fails, you may need to edit ```requirements.mac.txt``` and adjust version number to what is available. 
On newer M1 Macs, we also had issues with the h5py package, which could be solved by using brew for help (not sure this is needed any longer):

```sh
brew install hdf5
export HDF5_DIR="$(brew --prefix hdf5)"
pip3 install --no-binary=h5py h5py
```

You can also download all network checkpoint files (this should be done if you are installing for multiple users):
```sh
python3 FastSurferCNN/download_checkpoints.py --all
```

Once all dependencies are installed, run the FastSurfer segmentation only (!!) by calling ```bash ./run_fastsurfer.sh --seg_only ....``` with the appropriate command line flags, see the [README](README.md#usage). 

Note: You may always need to prepend the command with `bash` (i.e. `bash run_fastsurfer.sh <...>`) to ensure that bash 4.0 is used instead of the system default.

To run the full pipeline, install and source also the supported FreeSurfer version according to their [Instructions](https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads). There is a freesurfer email list, if you run into problems during this step.

#### 4. Apple AI Accelerator support
You can also try the experimental support for the Apple Silicon AI Accelerator by setting `PYTORCH_ENABLE_MPS_FALLBACK` and passing `--device mps`:

```sh
export PYTORCH_ENABLE_MPS_FALLBACK=1
./run_fastsurfer.sh --seg_only --device mps ....
```

This will be at least twice as fast as `--device cpu`. The fallback environment variable is necessary as one function is not yet implemented for the GPU and will fall back to CPU.

## Windows

### Docker (CPU version)

In order to run FastSurfer on your Windows system using docker make sure that you have:
* [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
* [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/)

installed and running.

After everything is installed, start Windows PowerShell and run the following command to pull the CPU Docker image (check on [dockerhub](https://hub.docker.com/r/deepmi/fastsurfer/tags) what version tag is most recent for cpu):

```bash
docker pull deepmi/fastsurfer:cpu-latest
```

Now you can run Fastsurfer the same way as described in our [README](README.md#example-1--fastsurfer-docker) for the CPU build, for example:
```bash
docker run -v C:/Users/user/my_mri_data:/data \
           -v C:/Users/user/my_fastsurfer_analysis:/output \
           -v C:/Users/user/my_fs_license_dir:/fs_license \
           --rm --user $(id -u):$(id -g) deepmi/fastsurfer:cpu-latest \
           --fs_license /fs_license/license.txt \
           --t1 /data/subjectX/orig.mgz \
           --device cpu \
           --sid subjectX --sd /output \
           --parallel
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

Now you can run Fastsurfer the same way as described in our [README](README.md#example-1--fastsurfer-docker), for example:
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

Note the [system requirements](https://github.com/Deep-MI/FastSurfer#system-requirements) of at least 8 GB system memory and 2 GB graphics memory for the GPU version. If the process fails, check if your [WSL2 distribution has enough memory reserved](https://www.aleksandrhovhannisyan.com/blog/limiting-memory-usage-in-wsl-2/).
