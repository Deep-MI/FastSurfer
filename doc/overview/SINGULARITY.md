# FastSurfer Singularity Support

## Sandbox


## 

`<*singularity-flags*>` includes flags that set up the singularity container:
- `--nv`: enable nVidia GPUs in Singularity (otherwise FastSurfer will run on the CPU),
- `-B <path>`: is used to share data between the host and Singularity (only paths listed here will be available to FastSurfer, see [Singularity documentation](doc/overview/README.md#sandbox) for more info).
  This should specifically include the "Subject Directory". If two paths are given like `-B /my/path/host:/other`, this means `/my/path/host/somefile` will be accessible inside Singularity in directory as `/other/somefile`.  
      


For use on HPCs (or in other cases where Docker is not available or preferred) you can easily create a Singularity image from the Docker image. 
Singularity uses its own image format, so the Docker images must be converted (we publish our releases as docker images available on [Dockerhub](https://hub.docker.com/r/deepmi/fastsurfer/tags)). 

## Singularity with the official FastSurfer Image
To create a Singularity image from the official FastSurfer image hosted on Dockerhub just run:
```bash
singularity build /path/to/fastsurfer-{{STABLE_VERSION}}.sif docker://deepmi/fastsurfer:cuda-v{{STABLE_VERSION}}
```
Singularity images are files - usually with the extension `.sif`. Here, we save the image in `/homer/user/my_singlarity_images`.
If you want to pick a specific FastSurfer version, you can also change the tag (`latest`) in `deepmi/fastsurfer:latest` to any tag. For example to use the cpu image hosted on [Dockerhub](https://hub.docker.com/r/deepmi/fastsurfer/tags) use the tag `cpu-latest`.

## Building your own FastSurfer Singularity Image
To build a custom FastSurfer Singularity image, the `Docker/build.py` script supports a flag for direct conversion.
Simply add `--singularity /path/to/singlarity_images/fastsurfer.sif` to the call, which first builds the image with Docker and then converts the image to Singularity.

If you want to manually convert the local Docker image `fastsurfer:myimage`, run:

```bash
singularity build /path/to/my/singlarity_images/fastsurfer-myimage.sif docker-daemon://fastsurfer:myimage
```

For more information on how to create your own Docker images, see our [Docker guide](../../Docker/README.md).

## FastSurfer Singularity Image Usage

After building the Singularity image, you need to register at the FreeSurfer website (https://surfer.nmr.mgh.harvard.edu/registration.html) to acquire a valid license (for free) - just as when using Docker. This license needs to be passed to the script via the `--fs_license` flag. This is not necessary if you want to run the segmentation only.

To run FastSurfer on a given subject using the Singularity image with GPU access, execute the following command:

```bash
singularity exec --nv \
                 --no-mount home,cwd \
                 -B /path/to/my/mri_data \
                 -B /path/to/my/fastsurfer_analysis \
                 -B /path/to/my/fs_license.txt \
                 /path/to/singlarity_images/fastsurfer.sif \
                 /fastsurfer/run_fastsurfer.sh \
                 --fs_license /path/to/my/fs_license.txt \
                 --t1 /path/to/my/mri_data/part1_image1.nii.gz \
                 --sid p001_img1 --sd /path/to/my/fastsurfer_analysis \
                 --parallel --3T
```
### Singularity Flags
* `--nv`: This flag is used to access GPU resources. It should be excluded if you intend to use the CPU version of FastSurfer
* `--no-mount home,cwd`: This flag tells singularity/apptainer to not mount the home directory and the current directory inside the singularity image (which it does by default, see [Best Practice](#mounting-home-and-working-directory))
* `-e`: This flag 
* `-B`: These commands mount your data, output, and directory with the FreeSurfer license file into the Singularity container. Inside the container these are visible under the name following the colon (in this case /data, /output, and /fs). 

### FastSurfer Flags
* The `--fs_license` points to your FreeSurfer license which needs to be available on your computer in the my_fs_license_dir that was mapped above, if you want to run the full surface analysis. 
* The `--t1` points to the t1-weighted MRI image to analyse (full path, with mounted name inside docker: /path/to/my/mri_data => /data)
* The `--sid` is the subject ID name (output folder name)
* The `--sd` points to the output directory (its mounted name inside docker: /path/to/my/fastsurfer_analysis => /output)
* The `--parallel` activates processing left and right hemisphere in parallel
* The `--3T` switches to the 3T atlas instead of the 1.5T atlas for Talairach registration. 

Note, that the paths following `--fs_license`, `--t1`, and `--sd` are __inside__ the container, not global paths on your system, so they should point to the places where you mapped these paths above with the `-B` arguments. 

A directory with the name as specified in `--sid` (here subjectX) will be created in the output directory. So in this example output will be written to /path/to/my/fastsurfer_analysis/subjectX/ . Make sure the output directory is empty, to avoid overwriting existing files. 

### Singularity without a GPU
You can run the Singularity equivalent of CPU-Docker by building a Singularity image from the CPU-Docker image (replace # with the current version number) and excluding the `--nv` argument in your Singularity exec command as following:

```bash
singularity build /path/to/singlarity_images/fastsurfer-cpu.sif docker://deepmi/fastsurfer:cpu-v{{STABLE_VERSION}}

singularity exec --no-home -e \
                 -B /path/to/my/mri_data:/data \
                 -B /path/to/my/fastsurfer_analysis:/output \
                 -B /path/to/my/fs_license_dir:/fs \
                 /path/to/singlarity_images/fastsurfer-cpu.sif \
                 /fastsurfer/run_fastsurfer.sh \
                 --fs_license /fs/license.txt \
                 --t1 /data/subjectX/orig.mgz \
                 --sid subjectX --sd /output \
                 --parallel --3T
```

## Singularity Best Practice

### Mounting Home and Working Directory
Mounting the user home directory into the container can have unexpected side effects, because some python packages and binaries read settings from the home directory. 
Therefore, it is recommended to **not mount the user home directory** into the singularity container as the home directory. 
This is achieved by `--no-mount home,cwd` (`--no-home` also avoids mounting of the user home directory, but apptainer also mounts the working directory, which may be home). Both of these are addressed by `--no-mount home,cwd`.

### Environment variables
Singularity also automatically keeps environment variables in the 

