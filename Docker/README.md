# FastSurfer Docker image creation

Within this directory we currently provide five different Dockerfiles that are set up for running: 

* the whole FastSurfer pipeline (FastSurferCNN + recon-surf, Example 1 (GPU) and 2 (CPU))
* only the segmentation network (FastSurferCNN, Example 3 (GPU) and 4 (CPU))
* only the surface pipeline (recon-surf, Example 5 (CPU))

In order to run the whole FastSurfer pipeline or the surface part, you need a valid FreeSurfer license (either from your local FreeSurfer installation or from the FreeSurfer website (https://surfer.nmr.mgh.harvard.edu/registration.html)). 

Note, in order to run our Docker containers on a Mac, users need to increase docker memory to 10 GB by overwriting the settings under Docker Desktop --> Preferences --> Resources --> Advanced (slide the bar under Memory to 10 GB; see: [docker for mac](https://docs.docker.com/docker-for-mac/) for details). Also for the new M1 Chip, try adding ´--platform linux/x86_64´ to the build and run commands below. 


### Example 1: Pull FastSurfer container 

We provide a number of prebuild docker images. In order to get them you simply need to execute the following command:

```bash 
docker pull deepmi/fastsurfer
```
By default, you fetch the deepmi/fastsurfer:latest image. You can get a different one by simply adding the corresponding tag at the end of "deepmi/fastsurfer" as in "deepmi/fastsurfer:latest". For a list of provided tags, visit our [Docker Hub](https://hub.docker.com/r/deepmi/fastsurfer/tags).

This script builds a docker image with the name fastsurfer:latest. With it, you basically execute the script __run_fastsurfer.sh__ from the parent directory. It takes as input a single T1-weighted MRI brain scan (from the /data directory) and first produces the aparc.DKTatlas+aseg.mgz segmentation followed by the surface construction (output stored in /output directory).

```bash
docker run --gpus all -v /home/user/my_mri_data:/data \
                      -v /home/user/my_fastsurfer_analysis:/output \
                      -v /home/user/my_fs_license_dir:/fs_license \
                      --rm --user $(id -u):$(id -g) deepmi/fastsurfer:latest \
                      --fs_license /fs_license/license.txt \
                      --t1 /data/subjectX/orig.mgz \
                      --sid subjectX --sd /output \
                      --parallel
```
##### Docker Flags:
* `--gpus`: This flag is used to access GPU resources. With it, you can also specify how many GPUs to use. In the example above, _all_ will use all available GPUS. To use a single one (e.g. GPU 0), set `--gpus device=0`. To use multiple specific ones (e.g. GPU 0, 1 and 3), set `--gpus "device=0,1,3"`.
* `-v`: This commands mount your data, output and directory with the FreeSurfer license file into the docker container. Inside the container these are visible under the name following the colon (in this case /data, /output, and /fs_license).
* `--rm`: The flag takes care of removing the container once the analysis finished. 
* `-d`: This is optional. You can add this flag to run in detached mode (no screen output and you return to shell)
* `--user $(id -u):$(id -g)`: This part automatically runs the container with your group- (id -g) and user-id (id -u). All generated files will then belong to the specified user. Without the flag, the docker container will be run as root.

##### Fastsurfer Flags:
* The `--fs_license` points to your FreeSurfer license which needs to be available on your computer in the my_fs_license_dir that was mapped above. 

Note, that the paths following `--fs_license`, `--t1`, and `--sd` are inside the container, not global paths on your system, so they should point to the places where you mapped these paths above with the `-v` arguments. 

A directory with the name as specified in `--sid` (here subjectX) will be created in the output directory. So in this example output will be written to /home/user/my_fastsurfer_analysis/subjectX/ . Make sure the output directory is empty, to avoid overwriting existing files. 

All other flags are identical to the ones explained on the main page [README](../README.md).

### Example 2: Build GPU FastSurfer container (default)

In order to build the docker image for FastSurfer (FastSurferCNN + recon-surf; on GPU; including FreeSurfer) yourself simply execute the following command after traversing into the *Docker* directory: 

```bash
cd ..
docker build --rm=true -t deepmi/fastsurfer:gpu-v2.0.0 -f ./Docker/Dockerfile .
```

For running the analysis, the command is basically the same as above for the prebuild option:
```bash
docker run --gpus all -v /home/user/my_mri_data:/data \
                      -v /home/user/my_fastsurfer_analysis:/output \
                      -v /home/user/my_fs_license_dir:/fs_license \
                      --rm --user $(id -u):$(id -g) deepmi/fastsurfer:gpu-v2.0.0 \
                      --fs_license /fs_license/license.txt \
                      --t1 /data/subjectX/orig.mgz \
                      --sid subjectX --sd /output \
                      --parallel
```

### Example 3: Build CPU FastSurfer container
In order to build the docker image for FastSurfer (FastSurferCNN + recon-surf; on CPU; including FreeSurfer) simply go to the parent directory (FastSurfer) and execute the docker build command directly:

```bash
cd ..
docker build --rm=true -t deepmi/fastsurfer:cpu-v2.0.0 -f ./Docker/Dockerfile_CPU .
```

For running the analysis, the command is basically the same as above for the GPU option:
```bash
docker run -v /home/user/my_mri_data:/data \
           -v /home/user/my_fastsurfer_analysis:/output \
           -v /home/user/my_fs_license_dir:/fs_license \
           --rm --user $(id -u):$(id -g) deepmi/fastsurfer:cpu-v2.0.0 \
           --fs_license /fs_license/license.txt \
           --t1 /data/subjectX/orig.mgz \
           --device cpu \
           --sid subjectX --sd /output \
           --parallel
```

As you can see, only the tag of the image is changed from gpu to cpu and the standard docker is used (no --gpus defined). In addition, the --device cpu flag is passed to explicitly turn on CPU usage inside FastSurferCNN.

### Example 4: Build GPU FastSurferCNN container (segmentation only)

In order to build the docker image for FastSurferCNN (segmentation only; on GPU; no FreeSurfer needed) simply go to the parent directory (FastSurfer) and execute the docker build command directly:

```bash
cd ..
docker build --rm=true -t deepmi/fastsurfer:gpu-segonly-v2.0.0 -f ./Docker/Dockerfile_FastSurferCNN .
```

For running the analysis, start the container (e.g. to run segmentation on __all__ subjects (scans named orig.mgz inside /home/user/my_mri_data/subjectX/mri/):
```bash
docker run --gpus all -v /home/user/my_mri_data:/data \
                      -v /home/user/my_fastsurferCNN_analysis:/output \
                      --rm --user $(id -u):$(id -g) deepmi/fastsurfer:gpu-segonly-v2.0.0 \
                      --t1 /data/subjectX/orig.mgz \
                      --sid subjectX --sd /output \
                      --seg_only
```

* The --gpus flag is used to access GPU resources. Specify how many GPUs to use: In the example above, _all_ will use all available GPUS. To use a single one (e.g. GPU 0), set --gpus device=0. To use multiple specific ones (e.g. GPU 0, 1 and 3), set --gpus '"device=0,1,3"'.
* The -v command mounts your data and output directory into the docker image. Inside it is visible under the name following the colon (in this case /data or /output).
* The --rm flag takes care of removing the container once the analysis finished. 
* Optionally, you can add a -d flag to run in detached mode (no screen output and you return to shell)
* Again, the --user $(id -u):$(id -g) part automatically runs the container with your group- _(id -g)_ and user-id _(id -u)_
* Also here, the paths after --i_dir and --o_dir refer to local paths inside the container, as they were mapped above with the -v commands.

##### Docker Flags:
* `--gpus`: This flag is used to access GPU resources. With it, you can also specify how many GPUs to use. In the example above, _all_ will use all available GPUS. To use a single one (e.g. GPU 0), set `--gpus device=0`. To use multiple specific ones (e.g. GPU 0, 1 and 3), set `--gpus "device=0,1,3"`.
* `-v`: This commands mount your data, output and directory with the FreeSurfer license file into the docker container. Inside the container these are visible under the name following the colon (in this case /data, /output, and /fs_license).
* `--rm`: The flag takes care of removing the container once the analysis finished. 
* `-d`: This is optional. You can add this flag to run in detached mode (no screen output and you return to shell)
* `--user $(id -u):$(id -g)`: This part automatically runs the container with your group- (id -g) and user-id (id -u). All generated files will then belong to the specified user. Without the flag, the docker container will be run as root.

Note, that the paths following `--t1`, and `--sd` are inside the container, not global paths on your system, so they should point to the places where you mapped these paths above with the `-v` arguments. 

A directory with the name as specified in `--sid` (here subjectX) will be created in the output directory. So in this example output will be written to /home/user/my_fastsurfer_analysis/subjectX/ . Make sure the output directory is empty, to avoid overwriting existing files. 

All other flags are identical to the ones explained on the main page [README](../README.md).




### Example 5: Build CPU FastSurferCNN container (segmentation only)
In order to build the docker image for FastSurferCNN (segmentation only; on CPU; no FreeSurfer needed) simply go to the parent directory (FastSurfer) and execute the docker build command directly:

```bash
cd ..
docker build --rm=true -t deepmi/fastsurfer:cpu-segonly-v2.0.0 -f ./Docker/Dockerfile_FastSurferCNN_CPU .
```

For running the analysis, start the container (e.g. to run segmentation on __all__ subjects (scans named orig.mgz inside /home/user/my_mri_data/subjectX/mri/):
```bash
docker run --gpus all -v /home/user/my_mri_data:/data \
                      -v /home/user/my_fastsurferCNN_analysis:/output \
                      --rm --user $(id -u):$(id -g) deepmi/fastsurfer:cpu-segonly-v2.0.0 \
                      --t1 /data/subjectX/orig.mgz \
                      --sid subjectX --sd /output \
                      --seg_only \
                      --device cpu
```

Again, only the tag of the image is changed from gpu to cpu and the standard docker is used (no --gpus defined). In addition, the --device cpu flag is passed to explicitly turn on CPU usage inside FastSurferCNN.

### Example 6: Build CPU FastSurfer recon-surf container (surface pipeline only)

In order to build the docker image for FastSurfer recon-surf (surface pipeline only, segmentation needs to exist already!) simply go to the parent directory (FastSurfer) and execute the docker build command directly:

```bash
cd ..
docker build --rm=true -t deepmi/fastsurfer:cpu-surfonly-v2.0.0 -f ./Docker/Dockerfile_reconsurf .
```

For running the analysis, start the container (e.g. to run segmentation on __all__ subjects (scans named orig.mgz inside /home/user/my_mri_data/subjectX/mri/):
```bash
docker run -v /home/user/my_mri_data:/data \
           -v /home/user/my_fastsurfer_analysis:/output \
           -v /home/user/my_fs_license_dir:/fs_license \
           --rm --user $(id -u):$(id -g) deepmi/fastsurfer:cpu-surfonly-v2.0.0 \
           --fs_license /fs_license/license.txt \
           --t1 /data/subjectX/orig.mgz \
           --sid subjectX --sd /output \
           --parallel \
           --surfonly
```
##### Docker Flags:
* `-v`: This commands mount your data, output and directory with the FreeSurfer license file into the docker container. Inside the container these are visible under the name following the colon (in this case /data, /output, and /fs_license).
* `--rm`: The flag takes care of removing the container once the analysis finished. 
* `--user $(id -u):$(id -g)`: This part automatically runs the container with your group- (id -g) and user-id (id -u). All generated files will then belong to the specified user. Without the flag, the docker container will be run as root.
* `-d`: This is optional. You can add this flag to run in detached mode (no screen output and you return to shell)

##### Fastsurfer Flags:
* The `--fs_license` points to your FreeSurfer license which needs to be available on your computer in the my_fs_license_dir that was mapped above. 

Note, that the paths following `--fs_license`, `--t1`, and `--sd` are inside the container, not global paths on your system, so they should point to the places where you mapped these paths above with the `-v` arguments. 

A directory with the name as specified in `--sid` (here subjectX) will be created in the output directory. So in this example output will be written to /home/user/my_fastsurfer_analysis/subjectX/ . Make sure the output directory is empty, to avoid overwriting existing files. 

All other flags are identical to the ones explained on the main page [README](../README.md).

### Example 7: Experimental build for AMD GPUs

Here we build an experimental image to test performance when running on AMD GPUs. Note that you need a supported OS and Kernel version and supported GPU for the RocM to work correctly. You need to install the Kernel drivers into 
your host machine kernel (amdgpu-install --usecase=dkms) for the amd docker to work. For this follow:
https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.2.3/page/Introduction_to_AMD_ROCm_Installation_Guide_for_Linux.html


```bash
cd ..
docker build --rm=true -t deepmi/fastsurfer:gpu-amd-v2.0.0 -f ./Docker/Dockerfile_FastSurferCNN_AMD .
```

and run segmentation only:

```bash
docker run --rm --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
           --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
	   --shm-size 8G \
	   -v /home/user/my_mri_data:/data \
	   -v /home/user/my_fastsurfer_analysis:/output \
	   deepmi/fastsurfer:gpu-amd-v2.0.0 \
	   --orig_name /data/subjectX/orig.mgz \
	   --pred_name /output/subjectX/aparc.DKTatlas+aseg.deep.mgz
```

Note, we tested on an AMD Radeon Pro W6600, which is not officially supported, but setting HSA_OVERRIDE_GFX_VERSION=10.3.0 inside docker did the trick.
