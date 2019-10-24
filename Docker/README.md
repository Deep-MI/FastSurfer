# FastSurfer Docker image creation

Within this directory we currently provide four different Dockerfiles that are set up for running either the whole FastSurfer pipeline (FastSurferCNN + recon-surf) or only the segmentation network (FastSurferCNN) on the GPU or CPU. 

### Example 1: Build GPU FastSurfer container (default)

In order to build the docker image for FastSurfer (FastSurferCNN + recon-surf; on GPU; including FreeSurfer) simply execute the following command after traversing into the *Docker* directory: 

```bash
./docker_build.sh
```

This script builds a docker image with the name fastsurfer:gpu. With it you basically execute the script __run_fastsurfer.sh__ from the parent directory. It takes as input a single T1-weighted MRI brain scan and first produces the aparc.DKTatlas+aseg.mgz segmentation followed by the surface construction.

```bash
nvidia-docker run -v /home/user/my_mri_data:/data --rm --user XXXX fastsurfer:gpu \
              --fs_license /data/.license \
              --t1 /data/subject2/orig.mgz \
              --seg /data/subject2/aparc.DKTatlas+aseg.deep.mgz \
              --sid subject8 --sd /data \
              --mc --qspec --nofsaparc --parallel
```

The fs_license points to your FreeSurfer license which needs to be available on your computer. The --user XXXX part should be changed to the appropiate user id.
The -v command mounts your data directory into the docker image. Inside it is visible under the name following the colon (in this case /data).
The --rm flag takes care of removing the container once the analysis finished. 
All other flags are identical to the ones explained on the main page (on directory up).

### Example 2: Build CPU FastSurfer container
In order to build the docker image for FastSurfer (FastSurferCNN + recon-surf; on CPU; including FreeSurfer) simply go to the parent directory (FastSurfer) and execute the docker build command directly:

```bash
cd ..
docker build -t fastsurfer:cpu -f ./Docker/Dockerfile_CPU .
```

For running the analysis, the command is basically the same as above for the GPU option:
```bash
docker run -v /home/user/my_mri_data:/data --rm --user XXXX fastsurfer:cpu \
              --fs_license /data/.license \
              --t1 /data/subject2/orig.mgz \
              --seg /data/subject2/aparc.DKTatlas+aseg.deep.mgz \
              --sid subject8 --sd /data \
              --mc --qspec --nofsaparc --parallel
```

As you can see, only the tag of the image is changed from gpu to cpu and the standard docker is used instead of the nvidia one.

### Build GPU FastSurferCNN container (segmentation only)

In order to build the docker image for FastSurferCNN (segmentation only; on GPU; no FreeSurfer needed) simply go to the parent directory (FastSurfer) and execute the docker build command directly:

```bash
cd ..
docker build -t fastsurfercnn:gpu -f ./Docker/Dockerfile_FastSurferCNN .
```

For running the analysis, start the container (e.g. to run segmentation on __all__ subjects (scans named orig.mgz inside /home/user/my_mri_data/subjectX/mri/):
```bash
nvidia-docker run -v /home/user/my_mri_data:/data --rm --user XXXX fastsurfercnn:gpu \
              --i_dir /data \
              --in_name mri/orig.mgz \
              --o_dir /data \
              --out_name aparc.DKTatlas+aseg.deep.mgz \
              --log deep_surfer.log
```

Again, the --user XXXX part should be changed to the appropiate user id.
The -v command mounts your data directory into the docker image. Inside it is visible under the name following the colon (in this case /data).
The --rm flag takes care of removing the container once the analysis finished. 
All other flags are identical to the ones explained on the main page (on directory up).

### Build CPU FastSurferCNN container (segmentation only)
In order to build the docker image for FastSurferCNN (segmentation only; on CPU; no FreeSurfer needed) simply go to the parent directory (FastSurfer) and execute the docker build command directly:

```bash
cd ..
docker build -t fastsurfercnn:cpu -f ./Docker/Dockerfile_FastSurferCNN_CPU .
```

For running the analysis, start the container (e.g. to run segmentation on __all__ subjects (scans named orig.mgz inside /home/user/my_mri_data/subjectX/mri/):
```bash
docker run -v /home/user/my_mri_data:/data --rm --user XXXX fastsurfercnn:cpu \
              --i_dir /data \
              --in_name mri/orig.mgz \
              --o_dir /data \
              --out_name aparc.DKTatlas+aseg.deep.mgz \
              --log deep_surfer.log
```

Again, only the tag of the image is changed from gpu to cpu and the standard docker is used instead of the nvidia one.