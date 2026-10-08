Singularity Support
===================

Containerization
----------------
Containerization tools like Apptainer (Singularity) or Docker provide several advantages.
Most importantly, they allow for exactly the same setup across different machines and even data centers and compute
clusters. They thus increase reproducibility by reducing software differences between evaluations.
Additionally, errors and unexpected behavior are easier to track down, since developers can reproduce the setup much
more easily.
Finally, containers provide a security advantage, because they can only access data that is explicitly shared with them,
which reduces the risk of both data theft and data encryption attacks. This strategy is also called
[sandboxing](https://en.wikipedia.org/wiki/Sandbox_(computer_security)).

Using Apptainer (or Singularity)
--------------------------------
In the following, we write "Singularity", but all steps work the same with the [open source Apptainer](https://apptainer.org).

To run FastSurfer in a Singularity container, you have to:
1. [download](#downloading-the-official-fastsurfer-image-for-singularity) or
   [create](#creating-your-own-fastsurfer-singularity-image) a Singularity image of FastSurfer.
2. [Start the Singularity container from the image](#starting-fastsurfer-from-a-singularity-image) with options for
   the container. It is useful to think of the image as a "hard drive" and the container as a "simulated computer
   inside the computer".

   We refer to these "options for the container" as `<singularity_flags>`. They are not options to FastSurfer (referred
   to as `<fastsurfer_flags>`), but to the "simulated computer", and define access to data, hardware (e.g. graphics
   cards), etc.

Downloading the official FastSurfer image for Singularity
---------------------------------------------------------
Singularity uses its own image format, so it downloads the official Docker images from
[Docker Hub](https://hub.docker.com/r/deepmi/fastsurfer/tags) and converts them.

To create an official FastSurfer Singularity image, run `singularity build`. Usage:
```text
singularity build <sif_path> <source>
```
For example:
```bash
singularity build $HOME/my_singularity_images/fastsurfer-{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}.sif \
    docker://deepmi/fastsurfer:{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}
```
Singularity images are files with the extension `.sif`. Here, we save the image in `$HOME/my_singularity_images`.
To use another image, change the tag `{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}` in `<source>`, for example to the
[CPU image](https://hub.docker.com/r/deepmi/fastsurfer/tags?name=cpu) (`cpu-v{{ FASTSURFER_VERSION }}`), to another
FastSurfer version, or to the image for another CUDA version, see
[which image fits your GPU](install/LINUX.md#nvidia-gpus).

Creating your own FastSurfer Singularity image
----------------------------------------------
To build a custom FastSurfer Singularity image, the `tools/Docker/build.py` script supports a flag for direct conversion.
Simply add `--singularity $HOME/my_singularity_images/fastsurfer-myimage.sif` to the call, which first builds the image
with Docker and then converts it to Singularity.

If you want to manually convert the local Docker image `fastsurfer:myimage`, run:

```bash
singularity build $HOME/my_singularity_images/fastsurfer-myimage.sif \
    docker-daemon://fastsurfer:myimage
```

For more information on how to create your own Docker images, see our [Docker guide](../../tools/Docker/README.md).

Starting FastSurfer from a Singularity image
-------------------------------------------
The surface reconstruction needs a FreeSurfer license, as with Docker:
[register at the FreeSurfer website](https://surfer.nmr.mgh.harvard.edu/registration.html) to get one for free, and pass
it to FastSurfer with the `--fs_license` flag. The segmentation alone does not need a license.

To run FastSurfer on a subject with the Singularity image and GPU access, execute:

```bash
freesurfer_license=${freesurfer_license:-/path/to/your/freesurfer/license_file}
singularity exec --nv \
                 --no-mount home,cwd -e \
                 -B $HOME/my_mri_data \
                 -B $HOME/my_fastsurfer_analysis \
                 -B $freesurfer_license \
                 $HOME/my_singularity_images/fastsurfer-{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}.sif \
                 /fastsurfer/run_fastsurfer.sh \
                 --fs_license $freesurfer_license \
                 --t1 $HOME/my_mri_data/subjectX/t1_weighted.nii.gz \
                 --sid subjectX --sd $HOME/my_fastsurfer_analysis \
                 --3T --threads 4
```
### Singularity Flags
* `--nv`: gives the container access to NVIDIA GPUs. Leave it out to run FastSurfer on the CPU.
* `--no-mount home,cwd`: tells Singularity not to mount the home directory or the current working directory inside the
  container (see [Best Practices](#best-practices)).
* `-e`: does not pass the environment variables of the host into the container.
* `-B <host_dir>`: shares a directory or file of the host with the container. Only paths listed here are available to
  FastSurfer, so these mount your data, the output directory and the FreeSurfer license file. Inside the container,
  they are visible under the same paths as on the host. With `-B <host_dir>:<container_dir>`, `<host_dir>/<file_name>`
  is visible inside the container as `<container_dir>/<file_name>` instead.

### FastSurfer Flags
* `--fs_license`: the path to your FreeSurfer license (needs to be shared with the container using `-B`).
* `--t1`: the path to the T1-weighted MRI image to analyze (needs to be shared with the container using `-B`).
* `--sid`: the subject ID (the name of the output folder).
* `--sd`: the path to the output directory (needs to be shared with the container using `-B`).
* `--3T`: uses the 3T atlas instead of the 1.5T atlas for the Talairach registration.

A directory with the name specified in `--sid` (here subjectX) will be created in the output directory, so in this
example, the output will be written to `$HOME/my_fastsurfer_analysis/subjectX/`. FastSurfer may overwrite files in
`$HOME/my_fastsurfer_analysis/subjectX/`.

### Singularity without a GPU
Without a GPU, build a Singularity image from the CPU image and leave out `--nv` in the `singularity exec` command:

```bash
freesurfer_license=${freesurfer_license:-/path/to/your/freesurfer/license_file}
singularity build $HOME/my_singularity_images/fastsurfer-cpu-v{{ FASTSURFER_VERSION }}.sif \
    docker://deepmi/fastsurfer:cpu-v{{ FASTSURFER_VERSION }}

singularity exec --no-mount home,cwd -e \
                 -B $HOME/my_mri_data \
                 -B $HOME/my_fastsurfer_analysis \
                 -B $freesurfer_license \
                 $HOME/my_singularity_images/fastsurfer-cpu-v{{ FASTSURFER_VERSION }}.sif \
                 /fastsurfer/run_fastsurfer.sh \
                 --fs_license $freesurfer_license \
                 --t1 $HOME/my_mri_data/subjectX/t1_weighted.nii.gz \
                 --sid subjectX --sd $HOME/my_fastsurfer_analysis \
                 --3T --threads 4
```

Common problems
---------------
1. Slow processing despite a GPU.

   FastSurfer runs on the CPU if it cannot use the GPU, and the log says why: for example, the container was started
   without `--nv`, or the NVIDIA driver is too old for the CUDA version of the image. Then update the driver or use the
   image for an older CUDA version, see [which image fits your GPU](install/LINUX.md#nvidia-gpus). If you built the
   underlying Docker image yourself, choose a different `--device` option.

2. Building a Singularity image from a local Docker image with
   `singularity build <sif_path> docker-daemon://fastsurfer:myimage` fails with an error message like this:
   ```text
   INFO:    Starting build...
   FATAL:   While performing build: conveyor failed to get: loading image from
     docker engine: Error response from daemon: {"message":"client version 1.22
     is too old. Minimum supported API version is 1.24, please upgrade your
     client to a newer version"}
   ```
   To solve this issue, export the image from Docker with `docker save -o <docker_archive_path> <image_tag>`, and build
   the Singularity image from that archive with `singularity build <sif_path> docker-archive:<docker_archive_path>`.

3. I get the following warning:
   ```text
   WARNING: Error changing the container working directory. Using '/' instead:
     chdir /home/***: no such file or directory
   ```
   This is because the home directory is not mounted inside the Singularity container (see
   [Best Practices](#best-practices)). You can ignore this warning, since `/` as the working directory does not cause any
   issues, or specify a different working directory with `--cwd <directory>`, for example `--cwd /fastsurfer`.

Best Practices
--------------

### Mounting Home and Current Working Directory
Do not mount the user home directory into the Singularity container as the home directory.

Why? If the user inside the Singularity container has access to a user directory, settings from that directory might
bleed into the FastSurfer pipeline.

How? Singularity mounts the home directory by default. To avoid this, specify `--no-mount home,cwd`. Additionally,
the `-e` flag ensures that no environment variables are passed from the host system into the container.
