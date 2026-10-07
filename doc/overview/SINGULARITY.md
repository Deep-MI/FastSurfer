Singularity Support
===================

Containerization
----------------
Containerization tools like Singularity, or Apptainer or Docker provide several advantages.
Most importantly, they allow for exactly same setup across different machines and even data centers and compute clusters. They thus increase reproducibility by reducing software differences between evaluations.
Additionally, errors and unexpected behavior is easier to track down, since the setup is significantly easier to reproduce for developers.
Finally, containers provide a security advantage, because the access to data is restricted to explicitly shared data reducing both the risk of data theft and data encryption attacks. This is strategy also called [sandboxing](https://en.wikipedia.org/wiki/Sandbox_(computer_security)).

Using Apptainer (or Singularity)
--------------------------------
In the following, we write "Singularity", but all steps work the same with the [open source Apptainer](https://apptainer.org).

To execute code in a Singularity container, users have to:
1. [download](SINGULARITY.md#downloading-the-official-fastsurfer-image-for-singularity) or [create](SINGULARITY.md#creating-your-own-fastsurfer-singularity-image) a Singularity image of FastSurfer.
2. [Start the Singularity container from a Singularity image](SINGULARITY.md#starting-fastsurfer-with-from-a-singularity-image) by defining options for the container. It is useful, to think of the image as a "hard drive" and the container as a "simulated computer inside the computer".

   We refer to these "options for the container" in `<singularity_flags>`. They are not options to FastSurfer (referred to as `<fastsurfer_flags>`), but to the "simulated computer" and define access to data, hardware (e.g. graphics cards), etc.

Downloading the official FastSurfer image for Singularity
---------------------------------------------------------
Singularity uses its own image format, so we need to download and convert the official docker images available from [Dockerhub](https://hub.docker.com/r/deepmi/fastsurfer/tags).

To create an official FastSurfer Singularity image, run `singularity build`. Usage:
```text
singularity build <sif_path> <source>
```
For example:
```bash
singularity build $HOME/my_singularity_images/fastsurfer-{{ FASTSURFER_VERSION }}.sif \
    docker://deepmi/fastsurfer:{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}
```
Singularity images are files with extension `.sif`. Here, we save the image in `$HOME/my_singularity_images`.
If you want to pick a specific FastSurfer version, you can also change `{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}` in the `<source>`. For example to use the [cpu image](https://hub.docker.com/r/deepmi/fastsurfer/tags?name=cpu) (`cpu-v{{ FASTSURFER_VERSION }}`) or a [specific CUDA version](https://hub.docker.com/r/deepmi/fastsurfer/tags?name=cu1) (check, which version is available the current FastSurfer version, for example `{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}`).

Creating your own FastSurfer Singularity image
----------------------------------------------
To build a custom FastSurfer Singularity image, the `tools/Docker/build.py` script supports a flag for direct conversion.
Simply add `--singularity $HOME/my_singularity_images/fastsurfer-myimage.sif` to the call, which first builds the image with Docker and then converts the image to Singularity.

If you want to manually convert the local Docker image `fastsurfer:myimage`, run:

```bash
singularity build $HOME/my_singularity_images/fastsurfer-myimage.sif \
    docker-daemon://fastsurfer:myimage
```

For more information on how to create your own Docker images, see our [Docker guide](../../tools/Docker/README.md).

Starting FastSurfer with from a Singularity image
-------------------------------------------------
After building the Singularity image, you need to [register at the FreeSurfer website](https://surfer.nmr.mgh.harvard.edu/registration.html) to acquire a valid license (for free) - just as when using Docker. This license needs to be passed to the script via the `--fs_license` flag. This is not necessary if you want to run the segmentation only.

To run FastSurfer on a given subject using the Singularity image with GPU access, execute the following command:

`<singularity_flags>` includes flags that set up the singularity container:
- `--nv`: enable nVidia GPUs in Singularity (otherwise FastSurfer will run on the CPU),
- `-B <host_dir>`: is used to share data between the host and Singularity (only paths listed here will be available to FastSurfer, see [Singularity documentation](SINGULARITY.md#containerization) for more info).
  This should specifically include the "Subject Directory". If two paths are given like `-B <host_dir>:<container_dir>`, this means `<host_dir>/<file_name>` will be accessible inside Singularity in directory as `<container_dir>/<file_name>`.

```bash
freesurfer_license=${freesurfer_license:-/path/to/your/freesurfer/license_file}
singularity exec --nv \
                 --no-mount home,cwd -e \
                 -B $HOME/my_mri_data \
                 -B $HOME/my_fastsurfer_analysis \
                 -B $freesurfer_license \
                  $HOME/my_singularity_images/fastsurfer-{{ FASTSURFER_VERSION }}.sif \
                  /fastsurfer/run_fastsurfer.sh \
                 --fs_license $freesurfer_license \
                 --t1 $HOME/my_mri_data/subjectX/t1_weighted.nii.gz \
                 --sid subjectX --sd $HOME/my_fastsurfer_analysis \
                 --3T --threads 4
```
### Singularity Flags
* `--nv`: This flag is used to access GPU resources. It should be excluded if you intend to use the CPU version of FastSurfer
* `-e`: Do not transfer the environment variables from the host to the container.
* `--no-mount home,cwd`: This flag tells singularity to not mount the home directory or the current working directory inside the singularity image (see [Best Practice](#best-practices))
* `-B`: These commands mount your data, output, and the FreeSurfer license file into the Singularity container. Inside the container these are visible under the same paths as on the host.

### FastSurfer Flags
* The `--fs_license` points to your FreeSurfer license (needs to be shared with the container using `-B`)
* The `--t1` points to the t1-weighted MRI image to analyse (needs to be shared with the container using `-B`)
* The `--sid` is the subject ID name (output folder name)
* The `--sd` points to the output directory (needs to be shared with the container using `-B`)
* The `--3T` switches to the 3T atlas instead of the 1.5T atlas for Talairach registration.

A directory with the name as specified in `--sid` (here subjectX) will be created in the output directory. So in this example output will be written to `$HOME/my_fastsurfer_analysis/subjectX/`. FastSurfer may overwrite files in `$HOME/my_fastsurfer_analysis/subjectX/`.

### Singularity without a GPU
You can run the Singularity equivalent of CPU-Docker by building a Singularity image from the CPU-Docker image (replace `{{ FASTSURFER_VERSION }}` with the version you want to use) and excluding the `--nv` argument in your Singularity exec command as following:

```bash
freesurfer_license=${freesurfer_license:-/path/to/your/freesurfer/license_file}
cd $HOME/my_singularity_images
singularity build fastsurfer-cpu-{{ FASTSURFER_VERSION }}.sif \
                  docker://deepmi/fastsurfer:cpu-v{{ FASTSURFER_VERSION }}

singularity exec --no-mount home,cwd -e \
                 -B $HOME/my_mri_data \
                 -B $HOME/my_fastsurfer_analysis \
                 -B $freesurfer_license \
                 $HOME/my_singularity_images/fastsurfer-cpu-{{ FASTSURFER_VERSION }}.sif \
                   /fastsurfer/run_fastsurfer.sh \
                     --fs_license $freesurfer_license \
                     --t1 $HOME/my_mri_data/subjectX/t1_weighted.nii.gz \
                     --sid subjectX --sd $HOME/my_fastsurfer_analysis \
                     --3T --threads 4
```

Common problems
---------------
1. Slow processing despite GPUs, log says `UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version ...)`.

   Your NVIDIA drivers are too old for the CUDA version used in the image you created, try using an image with an older CUDA version from [Docker Hub](https://hub.docker.com/r/deepmi/fastsurfer/tags), or specify a different `--device` option if you built the underlying Docker image yourself.

2. When building singularity image from the docker image via `singularity build <sif_path> docker-daemon://fastsurfer:myimage`, it may fail with an error message like this:
   ```text
   INFO:    Starting build...
   FATAL:   While performing build: conveyor failed to get: loading image from
     docker engine: Error response from daemon: {"message":"client version 1.22
     is too old. Minimum supported API version is 1.24, please upgrade your
     client to a newer version"}
   ```
   To solve this issue, you can export the image from docker with `docker save -o <docker_archive_path> <image_tag>` and then you can use singularity to build from that `singularity build <sif_path> docker-archive:<docker_archive_path>`.

3. I get the following warning:
   ```text
   WARNING: Error changing the container working directory. Using '/' instead:
     chdir /home/***: no such file or directory
   ```
   This is because the home directory is not mounted inside the singularity container (see [Best Practices](#best-practices)). You can ignore this warning, since `/` as the working directory does not cause any issues, or specify a different working directory with `--cwd <directory>`, for example `--cwd /fastsurfer`.

Best Practices
--------------

### Mounting Home and Current Working Directory
Do not mount the user home directory into the singularity container as the home directory.

Why? If the user inside the singularity container has access to a user directory, settings from that directory might bleed into the FastSurfer pipeline.

How? Singularity automatically mounts the home directory by default. To avoid this, specify `--no-mount home,cwd`. Additionally setting the `-e` flag will ensure that no environment variables will be passed from the host system into the container.
