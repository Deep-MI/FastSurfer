# FastSurfer Singularity Image Creation

We host our releases as docker images on [Dockerhub](https://hub.docker.com/r/deepmi/fastsurfer/tags)
For use on HPCs or in other cases where Docker is not preferred you can easily create a Singularity image from the Docker images. 

# FastSurfer Singularity Image Creation
For creating a singularity image from the Dockerhub just run: 

```bash
cd /home/user/my_singlarity_images
singularity build fastsurfer-latest.sif docker://deepmi/fastsurfer:latest
```

Singularity Images are saved as files. Here the _/homer/user/my_singlarity_images_ is the path where you want your file saved.
You can change _deepmi/fastsurfer:latest_ with any tag provided in our [Dockerhub](https://hub.docker.com/r/deepmi/fastsurfer/tags)

If you want to use a locally available image that you created yourself, instead run:

```bash
cd /home/user/my_singlarity_images
singularity build fastsurfer-myimage.sif docker-daemon://fastsurfer:myimage
```

For how to create your own Docker images see our [Docker guide](../Docker/README.md)

# FastSurfer Singularity Image Usage

After building the Singularity image, you need to register at the FreeSurfer website (https://surfer.nmr.mgh.harvard.edu/registration.html) to acquire a valid license (for free) - just as when using Docker. This license needs to be passed to the script via the `--fs_license` flag. This is not necessary if you want to run the segmentation only.

To run FastSurfer on a given subject using the Singularity image with GPU access, execute the following command:

```bash
singularity exec --nv -B /home/user/my_mri_data:/data \
                      -B /home/user/my_fastsurfer_analysis:/output \
                      -B /home/user/my_fs_license_dir:/fs \
                       /home/user/fastsurfer-gpu.sif \
                       /fastsurfer/run_fastsurfer.sh \
                      --fs_license /fs/license.txt \
                      --t1 /data/subjectX/orig.mgz \
                      --sid subjectX --sd /output \
                      --parallel
```
Singularity Flags:
* `--nv`: This flag is used to access GPU resources. It should be excluded if you intend to use the CPU version of FastSurfer
* `-B`: These commands mount your data, output, and directory with the FreeSurfer license file into the Singularity container. Inside the container these are visible under the name following the colon (in this case /data, /output, and /fs). 

FastSurfer Flags:
* The `--fs_license` points to your FreeSurfer license which needs to be available on your computer in the my_fs_license_dir that was mapped above, if you want to run the full surface analysis. 
* The `--t1` points to the t1-weighted MRI image to analyse (full path, with mounted name inside docker: /home/user/my_mri_data => /data)
* The `--sid` is the subject ID name (output folder name)
* The `--sd` points to the output directory (its mounted name inside docker: /home/user/my_fastsurfer_analysis => /output)
* The `--parallel` activates processing left and right hemisphere in parallel

Note, that the paths following `--fs_license`, `--t1`, and `--sd` are __inside__ the container, not global paths on your system, so they should point to the places where you mapped these paths above with the `-B` arguments. 

A directory with the name as specified in `--sid` (here subjectX) will be created in the output directory. So in this example output will be written to /home/user/my_fastsurfer_analysis/subjectX/ . Make sure the output directory is empty, to avoid overwriting existing files. 

You can run the Singularity equivalent of CPU-Docker by building a Singularity image from the CPU-Docker image (replace # with the current version number) and excluding the `--nv` argument in your Singularity exec command as following:

```bash
cd /home/user/my_singlarity_images
singularity build fastsurfer-gpu.sif docker://deepmi/fastsurfer:cpu-v#.#.#

singularity exec -B /home/user/my_mri_data:/data \
                 -B /home/user/my_fastsurfer_analysis:/output \
                 -B /home/user/my_fs_license_dir:/fs \
                  /home/user/fastsurfer-cpu.sif \
                  /fastsurfer/run_fastsurfer.sh \
                  --fs_license /fs/license.txt \
                  --t1 /data/subjectX/orig.mgz \
                  --sid subjectX --sd /output \
                  --parallel
```
