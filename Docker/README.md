# FastSurfer Docker image creation

Within this directory we currently provide five different Dockerfiles that are set up for running: 

* the whole FastSurfer pipeline (FastSurferCNN + recon-surf, Example 1 (GPU) and 2 (CPU))
* only the segmentation network (FastSurferCNN, Example 3 (GPU) and 4 (CPU))
* only the surface pipeline (recon-surf, Example 5 (CPU))

In order to run the whole FastSurfer pipeline or the surface part, you need a valid FreeSurfer license (either from your local FreeSurfer installation or from the FreeSurfer website (https://surfer.nmr.mgh.harvard.edu/registration.html)). 

Note, in order to run our Docker containers on a Mac, users need to increase docker memory to 10 GB by overwriting the settings under Docker Desktop --> Preferences --> Resources --> Advanced (slide the bar under Memory to 10 GB; see: [docker for mac](https://docs.docker.com/docker-for-mac/) for details). Also for the new M1 Chip, try adding ´--platform linux/x86_64´ to the build and run commands below. 

### Example 1: Build GPU FastSurfer container (default)

In order to build the docker image for FastSurfer (FastSurferCNN + recon-surf; on GPU; including FreeSurfer) simply execute the following command after traversing into the *Docker* directory: 

```bash
cd ..
docker build --rm=true -t fastsurfer:gpu -f ./Docker/Dockerfile .
```

This script builds a docker image with the name fastsurfer:gpu. With it you basically execute the script __run_fastsurfer.sh__ from the parent directory. It takes as input a single T1-weighted MRI brain scan (from the /data directory) and first produces the aparc.DKTatlas+aseg.mgz segmentation followed by the surface construction (output stored in /output directory).

```bash
docker run --gpus all -v /home/user/my_mri_data:/data \
                      -v /home/user/my_fastsurfer_analysis:/output \
                      -v /home/user/my_fs_license_dir:/fs60 \
                      --rm --user XXXX fastsurfer:gpu \
                      --fs_license /fs60/.license \
                      --t1 /data/subject2/orig.mgz \
                      --sid subject2 --sd /output \
                      --parallel
```

* The --gpus flag is used to access GPU resources. With it you can also specify how many GPUs to use. In the example above, _all_ will use all available GPUS. To use a single one (e.g. GPU 0), set --gpus device=0. To use multiple specific ones (e.g. GPU 0, 1 and 3), set --gpus '"device=0,1,3"'.
* The -v commands mount your data, output and directory with the FreeSurfer license file into the docker container. Inside the container these are visible under the name following the colon (in this case /data, /output, and /fs60).
* The --rm flag takes care of removing the container once the analysis finished. 
* The --user XXXX part should be changed to the appropriate user id (a four digit number; can be checked with the command "id -u" on linux systems). All generated files will then belong to the specified user. Without the flag, the docker container will be run as root.
* The fs_license points to your FreeSurfer license which needs to be available on your computer in the my_fs_license_dir that was mapped above. 
* Note, that the paths following --fs_license, --t1, and --sd are inside the container, not global paths on your system, so they should point to the places where you mapped these paths above with the -v arguments. 
* A directory with the name as specified in --sid (here subject2) will be created in the output directory. So in this example output will be written to /home/user/my_fastsurfer_analysis/subject2/ . Make sure the output directory is empty, to avoid overwriting existing files. 

All other flags are identical to the ones explained on the main page [README](../README.md).

### Example 2: Build CPU FastSurfer container
In order to build the docker image for FastSurfer (FastSurferCNN + recon-surf; on CPU; including FreeSurfer) simply go to the parent directory (FastSurfer) and execute the docker build command directly:

```bash
cd ..
docker build --rm=true -t fastsurfer:cpu -f ./Docker/Dockerfile_CPU .
```

For running the analysis, the command is basically the same as above for the GPU option:
```bash
docker run -v /home/user/my_mri_data:/data \
           -v /home/user/my_fastsurfer_analysis:/output \
           -v /home/user/my_fs_license_dir:/fs60 \
           --rm --user XXXX fastsurfer:cpu \
           --fs_license /fs60/.license \
           --t1 /data/subject2/orig.mgz \
           --no_cuda \
           --sid subject2 --sd /output \
           --parallel
```

As you can see, only the tag of the image is changed from gpu to cpu and the standard docker is used (no --gpus defined). In addition, the --no_cuda flag is passed to explicitly turn of GPU usage inside FastSurferCNN.

### Example 3: Build GPU FastSurferCNN container (segmentation only)

In order to build the docker image for FastSurferCNN (segmentation only; on GPU; no FreeSurfer needed) simply go to the parent directory (FastSurfer) and execute the docker build command directly:

```bash
cd ..
docker build --rm=true -t fastsurfercnn:gpu -f ./Docker/Dockerfile_FastSurferCNN .
```

For running the analysis, start the container (e.g. to run segmentation on __all__ subjects (scans named orig.mgz inside /home/user/my_mri_data/subjectX/mri/):
```bash
docker run --gpus all -v /home/user/my_mri_data:/data \
                      -v /home/user/my_fastsurferCNN_analysis:/output \
                      --rm --user XXXX fastsurfercnn:gpu \
                      --i_dir /data \
                      --in_name mri/orig.mgz \
                      --o_dir /output \
                      --out_name mri/aparc.DKTatlas+aseg.deep.mgz \
                      --log deep_surfer.log
```

* The --gpus flag is used to access GPU resources. Specify how many GPUs to use: In the example above, _all_ will use all available GPUS. To use a single one (e.g. GPU 0), set --gpus device=0. To use multiple specific ones (e.g. GPU 0, 1 and 3), set --gpus '"device=0,1,3"'.
* The -v command mounts your data and output directory into the docker image. Inside it is visible under the name following the colon (in this case /data or /output).
* The --rm flag takes care of removing the container once the analysis finished. 
* Again, the --user XXXX part should be changed to the appropiate user id (a four digit number; can be checked with the command "id -u" on linux systems).
* Also here, the paths after --i_dir and --o_dir refer to local paths inside the container, as they were mapped above with the -v commands.

All other flags are identical to the ones explained on the main page [README](../README.md).

### Example 4: Build CPU FastSurferCNN container (segmentation only)
In order to build the docker image for FastSurferCNN (segmentation only; on CPU; no FreeSurfer needed) simply go to the parent directory (FastSurfer) and execute the docker build command directly:

```bash
cd ..
docker build --rm=true -t fastsurfercnn:cpu -f ./Docker/Dockerfile_FastSurferCNN_CPU .
```

For running the analysis, start the container (e.g. to run segmentation on __all__ subjects (scans named orig.mgz inside /home/user/my_mri_data/subjectX/mri/):
```bash
docker run -v /home/user/my_mri_data:/data \
           -v /home/user/my_fastsurferCNN_analysis:/output \
           --rm --user XXXX fastsurfercnn:cpu \
           --i_dir /data \
           --in_name mri/orig.mgz \
           --o_dir /output \
           --out_name mri/aparc.DKTatlas+aseg.deep.mgz \
           --log deep_surfer.log \
           --no_cuda
```

Again, only the tag of the image is changed from gpu to cpu and the standard docker is used (no --gpus defined). In addition, the --no_cuda flag is passed to explicitly turn off GPU usage inside FastSurferCNN.

### Example 5: Build CPU FastSurfer recon-surf container (surface pipeline only)

In order to build the docker image for FastSurfer recon-surf (surface pipeline only, segmentation needs to exist already!) simply go to the parent directory (FastSurfer) and execute the docker build command directly:

```bash
cd ..
docker build --rm=true -t fastsurfer_reconsurf:cpu -f ./Docker/Dockerfile_reconsurf .
```

For running the analysis, start the container (e.g. to run segmentation on __all__ subjects (scans named orig.mgz inside /home/user/my_mri_data/subjectX/mri/):
```bash
docker run -v /home/user/my_mri_data:/data \
           -v /home/user/my_fastsurfer_analysis:/output \
           -v /home/user/my_fs_license_dir:/fs60 \
           --rm --user XXXX fastsurfer_reconsurf:cpu \
           --fs_license /fs60/.license \
           --t1 /data/subject2/orig.mgz \
           --sid subject2 --sd /output \
           --parallel
```
* The -v commands mount your data, output and directory with the FreeSurfer license file into the docker container. Inside the container these are visible under the name following the colon (in this case /data, /output, and /fs60).
* The --rm flag takes care of removing the container once the analysis finished. 
* Again, the --user XXXX part should be changed to the appropiate user id (a four digit number; can be checked with the command "id -u" on linux systems).
* The fs_license points to your FreeSurfer license which needs to be available on your computer in the my_fs_license_dir that was mapped above. 
* Note, that the paths following --fs_license, --t1, and --sd are inside the container, not global paths on your system, so they should point to the places where you mapped these paths above with the -v arguments. 

All other flags are identical to the ones explained on the main page [README](../README.md).
