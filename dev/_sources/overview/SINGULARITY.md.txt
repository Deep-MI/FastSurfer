Singularity Support
===================

Containerization
----------------
Containerization tools like Singularity, or Apptainer or Docker provide several advantages.
Most importantly, they allow for exactly same setup across different machines and even data centers and compute clusters. They thus increase reproducibility by reducing software differences between evaluations.
Additionally, errors and unexpected behavior is easier to track down, since the setup is significantly easier to reproduce for developers.
Finally, containers provide a security advantage, because the access to data is restricted to explicitly shared data reducing both the risk of data theft and data encryption attacks. This is strategy also called [sandboxing](https://en.wikipedia.org/wiki/Sandbox_(computer_security)).

Using Singularity (or Apptainer)
--------------------------------
In the following, we write "Singularity", but all steps work the same with the [open source Apptainer](https://apptainer.org).

To execute code in a Singularity container, users have to:
1. [download](SINGULARITY.md#downloading-the-official-fastsurfer-image-for-singularity) or [create](SINGULARITY.md#creating-your-own-fastsurfer-singularity-image) a Singularity image of FastSurfer.
2. [Start the Singularity container from a Singularity image](SINGULARITY.md#starting-fastsurfer-with-from-a-singularity-image) by defining options for the container. It is useful, to think of the image as a "hard drive" and the container as a "simulated computer inside the computer".

   We refer to these "options for the container" in `<*singularity-flags*>`. They are not options to FastSurfer (referred to as `<*fastsurfer-flags*>`), but to the "simulated computer" and define access to data, hardware (e.g. graphics cards), etc.

Downloading the official FastSurfer image for Singularity
---------------------------------------------------------
Singularity uses its own image format, so we need to download and convert the official docker images available from [Dockerhub](https://hub.docker.com/r/deepmi/fastsurfer/tags).

To create an official FastSurfer Singularity image, run:
```bash
# singularity build <filename> <source>
singularity build $HOME/my_singlarity_images/fastsurfer-2.5.0.sif docker://deepmi/fastsurfer:cuda-v2.5.0
```
Singularity images are files with extension `.sif`. Here, we save the image in `$HOME/my_singlarity_images`.
If you want to pick a specific FastSurfer version, you can also change `cuda-v2.5.0` in the `<source>`. For example to use the [cpu image](https://hub.docker.com/r/deepmi/fastsurfer/tags?name=cpu) (`cpu-v2.5.0`) or a [specific CUDA version](https://hub.docker.com/r/deepmi/fastsurfer/tags?name=cu1) (check, which version is available the current FastSurfer version, for example `cu118-v2.5.0`).

Creating your own FastSurfer Singularity image
----------------------------------------------
To build a custom FastSurfer Singularity image, the `Docker/build.py` script supports a flag for direct conversion.
Simply add `--singularity $HOME/my_singlarity_images/fastsurfer-myimage.sif` to the call, which first builds the image with Docker and then converts the image to Singularity.

If you want to manually convert the local Docker image `fastsurfer:myimage`, run:

```bash
singularity build $HOME/my_singlarity_images/fastsurfer-myimage.sif docker-daemon://fastsurfer:myimage
```

For more information on how to create your own Docker images, see our [Docker guide](../../tools/Docker/README.md).

Starting FastSurfer with from a Singularity image
-------------------------------------------------
After building the Singularity image, you need to [register at the FreeSurfer website](https://surfer.nmr.mgh.harvard.edu/registration.html) to acquire a valid license (for free) - just as when using Docker. This license needs to be passed to the script via the `--fs_license` flag. This is not necessary if you want to run the segmentation only.

To run FastSurfer on a given subject using the Singularity image with GPU access, execute the following command:

`<*singularity-flags*>` includes flags that set up the singularity container:
- `--nv`: enable nVidia GPUs in Singularity (otherwise FastSurfer will run on the CPU),
- `-B <path>`: is used to share data between the host and Singularity (only paths listed here will be available to FastSurfer, see [Singularity documentation](SINGULARITY.md#containerization) for more info).
  This should specifically include the "Subject Directory". If two paths are given like `-B /my/path/host:/other`, this means `/my/path/host/somefile` will be accessible inside Singularity in directory as `/other/somefile`.

```bash
singularity exec --nv \
                 --no-mount home,cwd -e \
                 -B $HOME/my_mri_data:/data \
                 -B $HOME/my_fastsurfer_analysis:/output \
                 -B $HOME/my_fs_license_dir:/fs \
                  $HOME/fastsurfer-gpu.sif \
                  /fastsurfer/run_fastsurfer.sh \
                 --fs_license /fs/license.txt \
                 --t1 /data/subjectX/orig.mgz \
                 --sid subjectX --sd /output \
                 --3T --threads 4
```
### Singularity Flags
* `--nv`: This flag is used to access GPU resources. It should be excluded if you intend to use the CPU version of FastSurfer
* `-e`: Do not transfer the environment variables from the host to the container.
* `--no-mount home,cwd`: This flag tells singularity to not mount the home directory or the current working directory inside the singularity image (see [Best Practice](#best-practices))
* `-B`: These commands mount your data, output, and directory with the FreeSurfer license file into the Singularity container. Inside the container these are visible under the name following the colon (in this case /data, /output, and /fs).

### FastSurfer Flags
* The `--fs_license` points to your FreeSurfer license (needs to be shared with the container using `-B`)
* The `--t1` points to the t1-weighted MRI image to analyse (needs to be shared with the container using `-B`)
* The `--sid` is the subject ID name (output folder name)
* The `--sd` points to the output directory (needs to be shared with the container using `-B`)
* The `--3T` switches to the 3T atlas instead of the 1.5T atlas for Talairach registration.

A directory with the name as specified in `--sid` (here subjectX) will be created in the output directory. So in this example output will be written to `$HOME/my_fastsurfer_analysis/subjectX/`. FastSurfer may overwrite files in `$HOME/my_fastsurfer_analysis/subjectX/`.

### Singularity without a GPU
You can run the Singularity equivalent of CPU-Docker by building a Singularity image from the CPU-Docker image (replace # with the current version number) and excluding the `--nv` argument in your Singularity exec command as following:

```bash
cd $HOME/my_singlarity_images
singularity build fastsurfer-cpu-2.5.0.sif docker://deepmi/fastsurfer:cpu-v2.5.0

singularity exec --no-mount -e \
                 -B $HOME/my_mri_data \
                 -B $HOME/my_fastsurfer_analysis \
                 -B $HOME/my_fs_license.txt \
                 $HOME/fastsurfer-cpu-{{ FASTSURFER_VERSION }}.sif \
                   /fastsurfer/run_fastsurfer.sh \
                     --fs_license $HOME/my_fs_license.txt \
                     --t1 $HOME/my_mri_data/subjectX/orig.mgz \
                     --sid subjectX --sd $HOME/my_fastsurfer_analysis \
                     --3T --threads 4
```

Common problems
---------------
1. Slow processing despite GPUs, log says `UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version ...)`.

   Your NVIDIA drivers are too old for the CUDA version used in the image you created, try using a different image with a different cuda version, for example for [CUDA 11](https://hub.docker.com/r/deepmi/fastsurfer/tags?name=cu11), or specify a different `--device` option if you built the underlying Docker image yourself.

2. When building singularity image from the docker image via `singularity build <singularity file location> docker-daemon://fastsurfer:myimage`, it may fail with an error message like this:
   ```
   INFO:    Starting build...
   FATAL:   While performing build: conveyor failed to get: loading image from docker engine: Error response from daemon: {"message":"client version 1.22 is too old. Minimum supported API version is 1.24, please upgrade your client to a newer version"}
   ```
   - To solve this issue, you can export the image from docker with `docker save -o <docker file location> <image tag>` and then you can use singularity to build from that `singularity build <singularity file name> docker-archive:<docker file location>`.

Best Practices
--------------

### Mounting Home and Current Working Directory
Do not mount the user home directory into the singularity container as the home directory.

Why? If the user inside the singularity container has access to a user directory, settings from that directory might bleed into the FastSurfer pipeline. For example, before FastSurfer 2.2 python packages installed in the user directory would replace those installed inside the image potentially causing incompatibilities. Since FastSurfer 2.2, `singularity exec ... --version +pip` outputs the FastSurfer version including a full list of python packages.

How? Singularity automatically mounts the home directory by default. To avoid this, specify `--no-mount home,cwd`. Additionally setting the `-e` flag will ensure that no environment variables will be passed from the host system into the container.
