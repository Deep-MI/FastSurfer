Running FastSurfer in a container
================================
This page explains the commands that run FastSurfer in our Apptainer (Singularity) or Docker images: the flags of the
container, and how FastSurfer's own flags work inside it. To get an image, see the installation page of your system
([Linux](install/LINUX.md), [Windows](install/WINDOWS.md), [Intel Macs](install/MACOS.md#docker-intel-macs)); the
[Examples](EXAMPLES.md) show complete commands for common tasks.

Containerization tools like Apptainer (Singularity) or Docker provide several advantages.
Most importantly, they allow for exactly the same setup across different machines and even data centers and compute
clusters. They thus increase reproducibility by reducing software differences between evaluations.
Additionally, errors and unexpected behavior are easier to track down, since developers can reproduce the setup much
more easily.
Finally, containers provide a security advantage, because they can only access data that is explicitly shared with them,
which reduces the risk of both data theft and data encryption attacks. This strategy is also called
[sandboxing](https://en.wikipedia.org/wiki/Sandbox_(computer_security)).

A container command has two parts: the options of the container (`<singularity_flags>` or `<docker_flags>`), which
define access to data and hardware such as graphics cards, and the options of FastSurfer (`<fastsurfer_flags>`), which
the container passes on to `run_fastsurfer.sh`. It is useful to think of the image as a "hard drive" and the container
as a "simulated computer inside the computer".

Using Apptainer (or Singularity)
--------------------------------
[Apptainer](https://apptainer.org) is the open-source continuation of Singularity, and both run the same images.
Apptainer also installs the `singularity` command, so the commands below use `singularity` and work with either.
[Linux](install/LINUX.md#apptainer-or-singularity) explains how to build the image (a `.sif` file).

To run FastSurfer on a subject with the Apptainer image and GPU access, execute:

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

Without a GPU, use the [CPU image](install/LINUX.md#cpu-only) and leave out `--nv`.

### Apptainer flags
* `--nv`: gives the container access to NVIDIA GPUs. Leave it out to run FastSurfer on the CPU.
* `--no-mount home,cwd`: tells Apptainer not to mount the home directory or the current working directory inside the
  container (see [below](#mounting-home-and-current-working-directory)).
* `-e`: does not pass the environment variables of the host into the container.
* `-B <host_dir>`: shares a directory or file of the host with the container. Only paths listed here are available to
  FastSurfer, so these mount your data, the output directory and the FreeSurfer license file. Inside the container,
  they are visible under the same paths as on the host. With `-B <host_dir>:<container_dir>`, `<host_dir>/<file_name>`
  is visible inside the container as `<container_dir>/<file_name>` instead.

### Mounting Home and Current Working Directory
Do not mount the user home directory into the Apptainer container as the home directory.

Why? If the user inside the Apptainer container has access to a user directory, settings from that directory might
bleed into the FastSurfer pipeline.

How? Apptainer mounts the home directory by default. To avoid this, specify `--no-mount home,cwd`. Additionally,
the `-e` flag ensures that no environment variables are passed from the host system into the container.

Without the home directory, Apptainer may warn:
```text
WARNING: Error changing the container working directory. Using '/' instead:
  chdir /home/***: no such file or directory
```
You can ignore this warning, since `/` as the working directory does not cause any issues, or specify a different
working directory with `--cwd <directory>`, for example `--cwd /fastsurfer`.

Using Docker
------------
[Linux](install/LINUX.md#docker) and [Windows](install/WINDOWS.md) explain how to download the image. To run
FastSurfer on a subject with the Docker image and GPU access, execute:

```bash
freesurfer_license=${freesurfer_license:-/path/to/your/freesurfer/license_file}
docker run --gpus all -v $HOME/my_mri_data:$HOME/my_mri_data \
           -v $HOME/my_fastsurfer_analysis:$HOME/my_fastsurfer_analysis \
           -v $freesurfer_license:$freesurfer_license \
           --rm --user $(id -u):$(id -g) deepmi/fastsurfer:{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }} \
           --fs_license $freesurfer_license \
           --t1 $HOME/my_mri_data/subjectX/t1_weighted.nii.gz \
           --sid subjectX --sd $HOME/my_fastsurfer_analysis \
           --3T --threads 4
```

Without a GPU, use the [CPU image](install/LINUX.md#cpu-only) and leave out `--gpus all`. AMD GPUs need
[additional flags](install/LINUX.md#amd-gpus-experimental).

### Docker flags
* `--gpus`: gives the container access to GPUs, and selects which ones. In the example above, `all` makes every GPU
  available to FastSurfer in the container. To use a single one (e.g. GPU 0), set `--gpus device=0`. To use several
  specific GPUs (e.g. GPU 0, 1 and 3), use `--gpus "device=0,1,3"`. Leave it out to run FastSurfer on the CPU.
* `-v`: defines which data is shared between the host system and the container, and how. By default, nothing is
  shared, so `-v` explicitly shares a directory or file. It follows the format `-v <host_dir>:<container_dir>:<options>`.
  In its simplest form, `<host_dir>` and `<container_dir>` are the same, so paths inside the container are the same as
  on the host. `:<options>` may be left out, or `:ro` makes the files read-only for the container. Share the input
  files, the output folder (subjects directory) and the FreeSurfer license.
* `--user $(id -u):$(id -g)`: the user and group the container runs as, which decides the access to files
  (**required**). `$(id -u)` and `$(id -g)` give your user and group IDs, so all generated files belong to you. Without
  this flag, FastSurfer exits with a message asking you to map your host user. Running the container as root
  (`--user 0:0`) is strongly discouraged and must be combined with the FastSurfer flag `--allow_root`.
* `--rm`: removes the container once the analysis has finished (optional, but recommended).
* `-d`: runs the container in detached mode, so you return to the shell without screen output (optional).
* `--group-add <group_list>`: if additional user groups are required to access files, add them with
  `--group-add <group_id>[,...]` or `--group-add $(id -G <group_name>)`.

Do not mount the user home directory into the Docker container as the home directory: settings from that directory
might bleed into the FastSurfer pipeline. Docker does not mount the home directory by default, so unless you manually
set the `HOME` environment variable, all should be fine.

FastSurfer flags in a container
-------------------------------
In principle, these are the same as for [run_fastsurfer.sh](../scripts/RUN_FASTSURFER.md#required-arguments). The
paths after `--fs_license`, `--t1` and `--sd` are paths __inside__ the container, so they have to be shared with `-B`
(Apptainer) or `-v` (Docker); in the examples above, they are the same as on your system.

* `--fs_license`: the path to your FreeSurfer license. In a container, it cannot be detected automatically and must
  be passed. The segmentation alone (`--seg_only`) does not need a license, see
  [FreeSurfer license](INSTALL.md#freesurfer-license).
* `--t1`: the path to the T1-weighted MRI image to analyze.
* `--sid`: the subject ID (the name of the output folder).
* `--sd`: the path to the output directory (the subjects directory).
* `--3T`: uses the 3T atlas instead of the 1.5T atlas for the Talairach registration, for better Talairach transforms
  and ICV estimates (eTIV).
* `--threads`: the number of threads for the segmentation and the surface reconstruction. With more than one thread,
  FastSurfer processes the left and right hemispheres in parallel. `max` uses all threads available, e.g. `16` on an
  8-core system with hyperthreading.

A directory with the name specified in `--sid` (here subjectX) will be created in the output directory, so in this
example, the output will be written to `$HOME/my_fastsurfer_analysis/subjectX/`. Make sure this directory does not
exist yet, to avoid overwriting existing files.

All other flags are explained in the [run_fastsurfer.sh documentation](../scripts/RUN_FASTSURFER.md).

Common problems
---------------
1. Slow processing despite a GPU.

   FastSurfer runs on the CPU if it cannot use the GPU, and the log says why: for example, the container was started
   without `--nv` or `--gpus all`, or the NVIDIA driver is too old for the CUDA version of the image. Then update the
   driver or use the image for an older CUDA version, see [which image fits your GPU](install/LINUX.md#nvidia-gpus).

2. FastSurfer stops with a message about the user (Docker).

   The container runs as the image's default user, so pass `--user $(id -u):$(id -g)`, see
   [Docker flags](#docker-flags).
