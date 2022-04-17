# FastSurfer Singularity image creation

After building a Docker image from the desired Dockerfile in ../Docker, you can build a Singularity image for use on HPCs or in other cases where Docker is not preferred.

In the absence of a Dockerhub image, you can push to a local Docker registry server:

```bash

#Start Docker registry for localhost
docker run -d -p 5000:5000 --restart=always --name registry registry:2

#tag your Docker image of FastSurfer and push to local registry
docker tag fastsurfer:gpu localhost:5000/fastsurfer:gpu
docker push localhost:5000/fastsurfer:gpu
```

Working from a directory in which you store Singularity images, you can then build your Singularity image:

```bash

singularity build fastsurfer-gpu.sif localhost:5000/fastsurfer:gpu
```

# FastSurfer Singularity image usage

After building the Singularity image, you need to register at the FreeSurfer website (https://surfer.nmr.mgh.harvard.edu/registration.html) to acquire a valid license (for free) - just as when using Docker. This license needs to be passed to the script via the --fs_license flag.

To run FastSurfer on a given subject using the Singularity image with GPU access, execute the following command:

```bash
singularity exec --nv -B /home/user/my_mri_data:/data \
                      -B /home/user/my_fastsurfer_analysis:/output \
                      -B /home/user/my_fs_license_dir:/fs60 \
                       /home/user/fastsurfer-gpu.sif \
                       /fastsurfer/run_fastsurfer.sh \
                      --fs_license /fs60/.license \
                      --t1 /data/subject2/orig.mgz \
                      --sid subject2 --sd /output \
                      --parallel
```

* The `--nv` flag is used to access GPU resources. This should be excluded if you intend to use the CPU version of FastSurfer
* The -B commands mount your data, output, and directory with the FreeSurfer license file into the Singularity container. Inside the container these are visible under the name following the colon (in this case /data, /output, and /fs60). 
* The fs_license points to your FreeSurfer license which needs to be available on your computer in the my_fs_license_dir that was mapped above. 
* Note, that the paths following --fs_license, --t1, and --sd are inside the container, not global paths on your system, so they should point to the places where you mapped these paths above with the -B arguments. 
* A directory with the name as specified in --sid (here subject2) will be created in the output directory. So in this example output will be written to /home/user/my_fastsurfer_analysis/subject2/ . Make sure the output directory is empty, to avoid overwriting existing files. 
* You can run the Singularity equivalent of CPU-Docker by building a Singularity image from the CPU-Docker image and excluding the `--nv` argument in your Singularity exec command.
