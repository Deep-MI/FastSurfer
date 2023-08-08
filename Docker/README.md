# Pull FastSurfer from DockerHub

We provide a number of prebuild docker images on [Docker Hub](https://hub.docker.com/r/deepmi/fastsurfer/tags). In order to get the latest GPU image you simply need to execute the following command:

```bash 
docker pull deepmi/fastsurfer
```
You can get a different one by simply adding the corresponding tag at the end of "deepmi/fastsurfer" as in "deepmi/fastsurfer:gpu-v#.#.#", where the # should be replaced with the version. 

After pulling the image, you can start a FastSurfer container and process a T1-weighted image (both segmentation and surface reconstruction) with the following command:

```bash
docker run --gpus all -v /home/user/my_mri_data:/data \
                      -v /home/user/my_fastsurfer_analysis:/output \
                      -v /home/user/my_fs_license_dir:/fs_license \
                      --rm --user $(id -u):$(id -g) deepmi/fastsurfer:latest \
                      --fs_license /fs_license/license.txt \
                      --t1 /data/subjectX/t1-weighted.nii.gz \
                      --sid subjectX --sd /output \
                      --parallel
```

##### Docker Flags:
* `--gpus`: This flag is used to access GPU resources. With it, you can also specify how many GPUs to use. In the example above, _all_ will use all available GPUS. To use a single one (e.g. GPU 0), set `--gpus device=0`. To use multiple specific ones (e.g. GPU 0, 1 and 3), set `--gpus "device=0,1,3"`.
* `-v`: This commands mount your data, output and directory with the FreeSurfer license file into the docker container. Inside the container these are visible under the name following the colon (in this case /data, /output, and /fs_license).
* `--rm`: The flag takes care of removing the container once the analysis finished. 
* `-d`: This is optional. You can add this flag to run in detached mode (no screen output and you return to shell)
* `--user $(id -u):$(id -g)`: This part automatically runs the container with your group- (id -g) and user-id (id -u). All generated files will then belong to the specified user. Without the flag, the docker container will be run as root which is strongly discouraged.

##### FastSurfer Flags:
* The `--fs_license` points to your FreeSurfer license which needs to be available on your computer in the my_fs_license_dir that was mapped above. 
* The `--t1` points to the t1-weighted MRI image to analyse (full path, with mounted name inside docker: /home/user/my_mri_data => /data)
* The `--sid` is the subject ID name (output folder name)
* The `--sd` points to the output directory (its mounted name inside docker: /home/user/my_fastsurfer_analysis => /output)
* The `--parallel` activates processing left and right hemisphere in parallel

Note, that the paths following `--fs_license`, `--t1`, and `--sd` are __inside__ the container, not global paths on your system, so they should point to the places where you mapped these paths above with the `-v` arguments. 

A directory with the name as specified in `--sid` (here subjectX) will be created in the output directory (specified via `--sd`). So in this example output will be written to /home/user/my_fastsurfer_analysis/subjectX/ . Make sure the output directory is empty, to avoid overwriting existing files. 

All other available flags are identical to the ones explained on the main page [README](../README.md).


# FastSurfer Docker Image Creation

Within this directory, we currently provide a build script and Dockerfile to create multiple Docker images for users (usually developers) who wish to create their own Docker images for 3 platforms:

* Nvidia / CUDA (Example 1)
* CPU (Example 2)
* AMD / rocm (experimental, Example 3)

To run only the surface pipeline or only the sgementatino pipeline, the entrypoint to these images has to be adapted, which is possible through
-  for the segmentation pipeline: `--entrypoint "python /fastsurfer/FastSurferCNN/run_prediction.py"`
-  for the surface pipeline: `--entrypoint "/fastsurfer/recon_surf/recon-surf.sh"`

Note, for many HPC users with limited GPUs or with very large datasets, it may be most efficient to run the full pipeline on the CPU, trading a longer run-time for the segmentation with massive parallelization on the subject level. 

Also note, in order to run our Docker containers on a Mac, users need to increase docker memory to 10 GB by overwriting the settings under Docker Desktop --> Preferences --> Resources --> Advanced (slide the bar under Memory to 10 GB; see: [docker for mac](https://docs.docker.com/docker-for-mac/) for details). For the new Apple silicon chips (M1,etc), we noticed that a native install runs much faster than docker when using the MPS device (experimental). 

### General build settings
The build script `build.py` supports additional args and targets and options, see `python Docker/build.py --help`.

Note, that the build script's main function is to select parameters for build args, but also create the FastSurfer-root/BUILD.info file, which will be used by FastSurfer to document the version (including git hash of the docker container). 
In general, if you specify `--dry_run` the command will not be executed but sent to stdout, so you can run `python build.py --device cuda --dry_run | bash` as well.

By default, the build script will tag your image as "fastsurfer:{version_tag}[-{device}]", where {version_tag} is {version-identifer from pyproject.toml}_{current git-hash} and {device} is the value to --device (and omitted for cuda), but a custom tag can be specified by `--tag {tag_name}`. 

### Example 1: Build GPU FastSurfer Image

In order to build your own Docker image for FastSurfer (FastSurferCNN + recon-surf; on GPU; including FreeSurfer) yourself simply execute the following command after traversing into the *Docker* directory: 

```bash
python build.py --device cuda --tag my_fastsurfer:cuda
```

For running the analysis, the command is basically the same as above for the prebuild option:
```bash
docker run --gpus all -v /home/user/my_mri_data:/data \
                      -v /home/user/my_fastsurfer_analysis:/output \
                      -v /home/user/my_fs_license_dir:/fs_license \
                      --rm --user $(id -u):$(id -g) my_fastsurfer:cuda \
                      --fs_license /fs_license/license.txt \
                      --t1 /data/subjectX/t1-weighted.nii.gz \
                      --sid subjectX --sd /output \
                      --parallel
```


### Example 2: Build CPU FastSurfer Image

In order to build the docker image for FastSurfer (FastSurferCNN + recon-surf; on CPU; including FreeSurfer) simply go to the parent directory (FastSurfer) and execute the docker build command directly:

```bash
python build.py --device cpu --tag my_fastsurfer:cpu
```

For running the analysis, the command is basically the same as above for the GPU option:
```bash
docker run -v /home/user/my_mri_data:/data \
           -v /home/user/my_fastsurfer_analysis:/output \
           -v /home/user/my_fs_license_dir:/fs_license \
           --rm --user $(id -u):$(id -g) my_fastsurfer:cpu \
           --fs_license /fs_license/license.txt \
           --t1 /data/subjectX/t1-weighed.nii.gz \
           --device cpu \
           --sid subjectX --sd /output \
           --parallel
```

As you can see, only the tag of the image is changed from gpu to cpu and the standard docker is used (no --gpus defined). In addition, the `--device cpu` flag is passed to explicitly turn on CPU usage inside FastSurferCNN.


### Example 3: Experimental Build for AMD GPUs

Here we build an experimental image to test performance when running on AMD GPUs. Note that you need a supported OS and Kernel version and supported GPU for the RocM to work correctly. You need to install the Kernel drivers into 
your host machine kernel (amdgpu-install --usecase=dkms) for the amd docker to work. For this follow:
https://docs.amd.com/en/latest/deploy/linux/quick_start.html

```bash
python build.py --device rocm --tag my_fastsurfer:rocm
```

and run segmentation only:

```bash
docker run --rm --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
           --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
	   --shm-size 8G \
	   -v /home/user/my_mri_data:/data \
	   -v /home/user/my_fastsurfer_analysis:/output \
	   my_fastsurfer:rocm \
	   --t1 /data/subjectX/t1-weighted.nii.gz \
	   --sid subjectX --sd /output 
```

Note, we tested on an AMD Radeon Pro W6600, which is [not officially supported](https://docs.amd.com/en/latest/release/gpu_os_support.html), but setting `HSA_OVERRIDE_GFX_VERSION=10.3.0` [inside docker did the trick](https://en.opensuse.org/AMD_OpenCL#ROCm_-_Running_on_unsupported_hardware):

```bash
docker run --rm --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
           --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
	   --shm-size 8G \
	   -v /home/user/my_mri_data:/data \
	   -v /home/user/my_fastsurfer_analysis:/output \
	   -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
	   my_fastsurfer:rocm \
	   --t1 /data/subjectX/t1-weighted.nii.gz \
	   --sid subjectX --sd /output 
```

