# FastSurfer Docker image creation

Within this directory we currently provide five different Dockerfiles that are set up for running: 

* the whole FastSurfer pipeline (FastSurferCNN + recon-surf, Example 1 (GPU) and 2 (CPU))
* only the segmentation network (FastSurferCNN, Example 3 (GPU) and 4 (CPU))
* only the surface pipeline (recon-surf, Example 5 (CPU))

In order to run the whole FastSurfer pipeline or the surface part, you need you need a valid FreeSurfer license (either from your local FreeSurfer installation or from the FreeSurfer website (https://surfer.nmr.mgh.harvard.edu/registration.html)). 

### Example 1: Build GPU FastSurfer container (default)

In order to build the docker image for FastSurfer (FastSurferCNN + recon-surf; on GPU; including FreeSurfer) simply execute the following command after traversing into the *Docker* directory: 

```bash
./docker_build.sh
```

This script builds a docker image with the name fastsurfer:gpu. With it you basically execute the script __run_fastsurfer.sh__ from the parent directory. It takes as input a single T1-weighted MRI brain scan (from the /data directory) and first produces the aparc.DKTatlas+aseg.mgz segmentation followed by the surface construction (output stored in /output directory).

```bash
docker run --gpus all -v /home/user/my_mri_data:/data \
                      -v /home/user/my_fastsurfer_analysis:/output \
                      -v /home/user/my_fs_license_dir:/fs60 \
                      --rm --user XXXX fastsurfer:gpu \
                      --fs_license /fs60/.license \
                      --t1 /data/subject2/orig.mgz \
                      --seg /output/subject2/aparc.DKTatlas+aseg.deep.mgz \
                      --sid subject2 --sd /output \
                      --mc --qspec --nofsaparc --parallel
```

* The fs_license points to your FreeSurfer license which needs to be available on your computer (e.g. in the /home/user/my_fs_license_dir folder). 
* The --gpus flag is used to access GPU resources. With it you can also specify how many GPUs to use. In the example above, _all_ will use all available GPUS. To use a single one (e.g. GPU 0), set --gpus device=0. To use multiple specific ones (e.g. GPU 0, 1 and 3), set --gpus '"device=0,1,3"'.
* The -v command mounts your data (and output) directory into the docker image. Inside it is visible under the name following the colon (in this case /data or /output).
* The --rm flag takes care of removing the container once the analysis finished. 
* The --user XXXX part should be changed to the appropriate user id (a four digit number; can be checked with the command "id -u" on linux systems). All generated files will then belong to the specified user. Without the flag, the docker container will be run as root.

All other flags are identical to the ones explained on the main page (on directory up).

### Example 2: Build CPU FastSurfer container
In order to build the docker image for FastSurfer (FastSurferCNN + recon-surf; on CPU; including FreeSurfer) simply go to the parent directory (FastSurfer) and execute the docker build command directly:

```bash
cd ..
docker build -t fastsurfer:cpu -f ./Docker/Dockerfile_CPU .
```

For running the analysis, the command is basically the same as above for the GPU option:
```bash
docker run -v /home/user/my_mri_data:/data \
           -v /home/user/my_fastsurfer_analysis:/output \
           -v /home/user/my_fs_license_dir:/fs60 \
           --rm --user XXXX fastsurfer:cpu \
           --fs_license /fs60/.license \
           --t1 /data/subject2/orig.mgz \
           --seg /output/subject2/aparc.DKTatlas+aseg.deep.mgz \
           --no_cuda \
           --sid subject2 --sd /output \
           --mc --qspec --nofsaparc --parallel
```

As you can see, only the tag of the image is changed from gpu to cpu and the standard docker is used (no --gpus defined). In addition, the --no_cuda flag is passed to explicitly turn of GPU usage inside FastSurferCNN.

### Example 3: Build GPU FastSurferCNN container (segmentation only)

In order to build the docker image for FastSurferCNN (segmentation only; on GPU; no FreeSurfer needed) simply go to the parent directory (FastSurfer) and execute the docker build command directly:

```bash
cd ..
docker build -t fastsurfercnn:gpu -f ./Docker/Dockerfile_FastSurferCNN .
```

For running the analysis, start the container (e.g. to run segmentation on __all__ subjects (scans named orig.mgz inside /home/user/my_mri_data/subjectX/mri/):
```bash
docker run --gpus all -v /home/user/my_mri_data:/data \
                      -v /home/user/my_fastsurferCNN_analysis:/output \
                      --rm --user XXXX fastsurfercnn:gpu \
                      --i_dir /data \
                      --in_name mri/orig.mgz \
                      --o_dir /output \
                      --out_name aparc.DKTatlas+aseg.deep.mgz \
                      --log deep_surfer.log
```

* The --gpus flag is used to access GPU resources. Specify how many GPUs to use: In the example above, _all_ will use all available GPUS. To use a single one (e.g. GPU 0), set --gpus device=0. To use multiple specific ones (e.g. GPU 0, 1 and 3), set --gpus '"device=0,1,3"'.
* The -v command mounts your data and output directory into the docker image. Inside it is visible under the name following the colon (in this case /data or /output).
* The --rm flag takes care of removing the container once the analysis finished. 
* Again, the --user XXXX part should be changed to the appropiate user id (a four digit number; can be checked with the command "id -u" on linux systems).

All other flags are identical to the ones explained on the main page (on directory up).

### Example 4: Build CPU FastSurferCNN container (segmentation only)
In order to build the docker image for FastSurferCNN (segmentation only; on CPU; no FreeSurfer needed) simply go to the parent directory (FastSurfer) and execute the docker build command directly:

```bash
cd ..
docker build -t fastsurfercnn:cpu -f ./Docker/Dockerfile_FastSurferCNN_CPU .
```

For running the analysis, start the container (e.g. to run segmentation on __all__ subjects (scans named orig.mgz inside /home/user/my_mri_data/subjectX/mri/):
```bash
docker run -v /home/user/my_mri_data:/data \
           -v /home/user/my_fastsurferCNN_analysis:/output \
           --rm --user XXXX fastsurfercnn:cpu \
           --i_dir /data \
           --in_name mri/orig.mgz \
           --o_dir /output \
           --out_name aparc.DKTatlas+aseg.deep.mgz \
           --log deep_surfer.log \
           --no_cuda
```

Again, only the tag of the image is changed from gpu to cpu and the standard docker is used (no --gpus defined). In addition, the --no_cuda flag is passed to explicitly turn of GPU usage inside FastSurferCNN.

### Example 5: Build CPU FastSurfer recon-surf container (surface pipeline only)

In order to build the docker image for FastSurfer recon-surf (surface pipeline only, segmentation needs to exist already!) simply go to the parent directory (FastSurfer) and execute the docker build command directly:

```bash
cd ..
docker build -t fastsurfer_reconsurf:cpu -f ./Docker/Dockerfile_reconsurf .
```

For running the analysis, start the container (e.g. to run segmentation on __all__ subjects (scans named orig.mgz inside /home/user/my_mri_data/subjectX/mri/):
```bash
docker run -v /home/user/my_mri_data:/data \
           -v /home/user/my_fastsurfer_analysis:/output \
           -v /home/user/my_fs_license_dir:/fs60 \
           --rm --user XXXX fastsurfer_reconsurf:cpu \
           --fs_license /fs60/.license \
           --t1 /data/subject2/orig.mgz \
           --seg /output/subject2/mri/aparc.DKTatlas+aseg.deep.mgz \
           --sid subject2 --sd /output \
           --mc --qspec --nofsaparc --parallel
```
* The fs_license points to your FreeSurfer license which needs to be available on your computer (e.g. in the /home/user/my_fs_license_dir folder). 
* The -v command mounts your data and output directory into the docker image. Inside it is visible under the name following the colon (in this case /data or /output).
* The --rm flag takes care of removing the container once the analysis finished. 
* Again, the --user XXXX part should be changed to the appropiate user id (a four digit number; can be checked with the command "id -u" on linux systems).

All other flags are identical to the ones explained on the main page (on directory up).
