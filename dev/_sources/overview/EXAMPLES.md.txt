Examples
========
Example 1: FastSurfer Singularity (or Apptainer)
------------------------------------------------
Singularity and Apptainer are alternative containerization solutions. Both have open-source distributions and are often
available in HPC settings. See our [Singularity docs](SINGULARITY.md) for more details.

### Preparation
Build the Singularity image (see below or [these instructions](SINGULARITY.md)). If you intend to generate surfaces,
you need to [register at the FreeSurfer website](https://surfer.nmr.mgh.harvard.edu/registration.html) to acquire a
FreeSurfer license (for free). This license needs to be passed to FastSurfer via the `--fs_license` flag. If you do not
intend to generate surfaces, it is often not necessary to obtain a FreeSurfer license.

```bash
# Build the singularity image (if it does not exist)
singularity build fastsurfer-gpu.sif docker://deepmi/fastsurfer
```

### Running FastSurfer
To run FastSurfer on a given subject using the Singularity image with GPU access, execute the following commands from a
directory where you want to store singularity images. This will create a singularity image from our Dockerhub image and
execute it:

```bash
singularity exec --nv \
                 --no-mount home,cwd -e \
                 -B $HOME/my_mri_data:$HOME/my_mri_data \
                 -B $HOME/my_fastsurfer_analysis:$HOME/my_fastsurfer_analysis \
                 -B $HOME/my_fs_license.txt:$HOME/my_fs_license.txt \
                 ./fastsurfer-gpu.sif \
                   /fastsurfer/run_fastsurfer.sh \
                     --fs_license $HOME/my_fs_license.txt \
                     --t1 $HOME/my_mri_data/subjectX/t1-weighted.nii.gz \
                     --sid subjectX --sd $HOME/my_fastsurfer_analysis \
                     --3T \
                     --threads 4
```

### Singularity Flags
* The `--nv` flag is used to access GPU resources.
* The `--no-home` flag stops mounting your home directory into singularity.
* The `-B` commands mount your data, output, and directory with the FreeSurfer license file into the Singularity container. Inside the container these are visible under the name following the colon (in this case /data, /output, and /fs_license).

### FastSurfer Flags
* The `--fs_license` points to your FreeSurfer license which needs to be available on your computer in the my_fs_license_dir that was mapped above.
* The `--t1` points to the t1-weighted MRI image to analyse (full path, must be mounted via `-B`)
* The `--sid` is the subject ID name (output folder name)
* The `--sd` points to the output directory (must be mounted via `-B`)
* The `--3T` changes the atlas for registration to the 3T atlas for better Talairach transforms and ICV estimates (eTIV)
* The `--threads` tells FastSurfer to use that many threads in segmentation and surface reconstruction. `max` will auto-detect the number of threads available, i.e. `16` on an 8-core system with hypterthreading. If the number of threads is greater than 1, FastSurfer will process the left and right hemispheres in parallel.

Note, that the paths following `--fs_license`, `--t1`, and `--sd` are __inside__ the container, not global paths on your system, so they should point to the places where you mapped these paths above with the `-v` arguments (part after colon).

A directory with the name as specified in `--sid` (here subjectX) will be created in the output directory. So in this example output will be written to `$HOME/my_fastsurfer_analysis/subjectX/` . Make sure the output directory is empty, to avoid overwriting existing files.

If you have no supported GPU, most Singularity images should automatically work (default to the CPU, just drop the `--nv` flag). Since execution on the CPU requires less driver installation, a custom, smaller CPU image is available `singularity build fastsurfer-cpu.sif docker://deepmi/fastsurfer:cpu-latest`.

Example 2: FastSurfer Docker
----------------------------
After pulling one of our images from Dockerhub, you do not need to have a separate installation of FreeSurfer on your computer (it is already included in the Docker image). However, if you want to run ___more than just the segmentation CNN___, you need to [register at the FreeSurfer website](https://surfer.nmr.mgh.harvard.edu/registration.html) to acquire a valid license for free. The directory containing the license needs to be mounted and passed to the script via the `--fs_license` flag. Basically for Docker (as for Singularity below) you are starting a container image (with the run command) and pass several parameters for that, e.g. if GPUs will be used and mounting (linking) the input and output directories to the inside of the container image. In the second half of that call you pass parameters to our `run_fastsurfer.sh` script that runs inside the container (e.g. where to find the FreeSurfer license file, and the input data and other flags).

To run FastSurfer on a given subject using the provided GPU-Docker, execute the following command:

```bash
docker run --gpus all -v $HOME/my_mri_data:$HOME/my_mri_data \
                      -v $HOME/my_fastsurfer_analysis:$HOME/my_fastsurfer_analysis \
                      -v $HOME/my_fs_license.txt:$HOME/my_fs_license.txt \
                      --rm --user $(id -u):$(id -g) deepmi/fastsurfer:latest \
                      --fs_license $HOME/my_fs_license.txt \
                      --t1 $HOME/my_mri_data/subjectX/t1-weighted.nii.gz \
                      --sid subjectX --sd $HOME/my_fastsurfer_analysis \
                      --3T \
                      --threads 4
```

### Docker Flags
* The `--gpus` flag is used to allow Docker to access GPU resources. With it, you can also specify how many GPUs to use. In the example above, _all_ will use all available GPUS. To use a single one (e.g. GPU 0), set `--gpus device=0`. To use multiple specific ones (e.g. GPU 0, 1 and 3), set `--gpus 'device=0,1,3'`. If you do not have a supported GPU, just drop this flag to use the CPU.
* The `-v` commands mount your data, output, and directory with the FreeSurfer license file into the docker container. Inside the container these are visible under the name following the colon (in this case /data, /output, and /fs_license).
* The `--rm` flag takes care of removing the container once the analysis finished.
* The `--user $(id -u):$(id -g)` part automatically runs the container with your group- (`id -g`) and user-id (`id -u`). All generated files will then belong to the specified user. Without the flag, the docker container will return an error. If running the container as root is required (despite being against best practice, for example because it is run in a sandbox, pass `--user 0:0`).

### Docker image
* This command assumes you want to use the most recent (locally cached) version of FastSurfer `deepmi/fastsurfer:latest`. This will always include current nVidia drivers and libraries.
* For older libraries, an image with AMD drivers or a smaller, CPU-only docker image, images are available in [multiple configurations](https://hub.docker.com/r/deepmi/fastsurfer/tags).

### FastSurfer Flag
* The `--fs_license` points to your FreeSurfer license which needs to be available on your computer, replace all occurrences of `$HOME/my_fs_license.txt` (full path, must be mounted via `-v <path>:<path>`).
* The `--t1` points to the t1-weighted MRI image to analyse (full path, must be mounted via `-v <path>:<path>`)
* The `--sid` is the subject ID name (output folder name)
* The `--sd` points to the output directory (must be mounted via `-v <path>:<path>`)
* The `--3T` changes the atlas for registration to the 3T atlas for better Talairach transforms and ICV estimates (eTIV)
* The `--threads` tells FastSurfer to use that many threads in segmentation and surface reconstruction. `max` will auto-detect the number of threads available, i.e. `16` on an 8-core system with hyperthreading. If the number of threads is greater than 1, FastSurfer will process the left and right hemispheres in parallel.

A directory with the name as specified in `--sid` (here subjectX) will be created in the output directory if it does not exist. So in this example output will be written to `$HOME/my_fastsurfer_analysis/subjectX/`. Make sure the output directory is empty, to avoid overwriting existing files.

Example 3: Native FastSurfer on subjectX with parallel processing of hemis
--------------------------------------------------------------------------
For a native install you may want to make sure that you are on our stable branch, as the default dev branch is for development and could be broken at any time. For that you can directly clone the stable branch:

```bash
git clone --branch stable https://github.com/Deep-MI/FastSurfer.git
```

More details (e.g. you need all dependencies in the right versions and also FreeSurfer locally) can be found in our [Installation guide](INSTALL.md).
Given you want to analyze data for subject which is stored on your computer under `$HOME/my_mri_data/subjectX/t1-weighted.nii.gz`, run the following command from the console (do not forget to source FreeSurfer!):

```bash
# Source FreeSurfer
export FREESURFER_HOME=/path/to/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Define data directory
datadir=$HOME/my_mri_data
fastsurferdir=$HOME/my_fastsurfer_analysis

# Run FastSurfer
./run_fastsurfer.sh --t1 $datadir/subjectX/t1-weighted-nii.gz \
                    --sid subjectX --sd $fastsurferdir \
                    --threads 4 --3T
```

The output will be stored in the `$fastsurferdir` (including the `aparc.DKTatlas+aseg.deep.mgz` segmentation under `$fastsurferdir/subjectX/mri` (default location)). Processing of the hemispheres will be run in parallel (`--threads 4`, 4 >= 2) to significantly speed-up surface creation. Omit this flag to run the processing sequentially, e.g. if you want to save resources on a compute cluster.


Example 4: FastSurfer on multiple subjects
------------------------------------------
In order to run FastSurfer on multiple cases, you may use the helper script `brun_subjects.sh`. This script accepts multiple ways to define the subjects, for example a subjects_list file.
Prepare the subjects_list file as follows (one line subject per line; delimited by `\n`):
```
subject_id1=path_to_t1
subject2=path_to_t1
subject3=path_to_t1
...
subject10=path_to_t1
```
Note, that all paths (`path_to_t1`) are as if you passed them to the `run_fastsurfer.sh` script via `--t1 <path>` so they may be with respect to the singularity or docker file system. Absolute paths are recommended.

The `brun_fastsurfer.sh` script can then be invoked in docker, singularity or on the native platform as follows:

### Docker
```bash
docker run --gpus all -v $HOME/my_mri_data:/data \
                      -v $HOME/my_fastsurfer_analysis:/output \
                      -v $HOME/my_fs_license_dir:/fs_license \
                      --entrypoint "/fastsurfer/brun_fastsurfer.sh" \
                      --rm --user $(id -u):$(id -g) deepmi/fastsurfer:latest \
                      --fs_license /fs_license/license.txt \
                      --sd /output --subject_list /data/subjects_list.txt \
                      --3T \
                      --threads 4
```
### Singularity
```bash
singularity exec --nv \
                 --no-home \
                 -B $HOME/my_mri_data:/data \
                 -B $HOME/my_fastsurfer_analysis:/output \
                 -B $HOME/my_fs_license_dir:/fs_license \
                 ./fastsurfer-gpu.sif \
                 /fastsurfer/brun_fastsurfer.sh \
                 --fs_license /fs_license/license.txt \
                 --sd /output \
                 --subject_list /data/subjects_list.txt \
                 --3T \
                 --threads 4
```
### Native
```bash
export FREESURFER_HOME=/path/to/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

cd $HOME/FastSurfer
datadir=$HOME/my_mri_data
fastsurferdir=$HOME/my_fastsurfer_analysis

# Run FastSurfer
./brun_fastsurfer.sh --subject_list $datadir/subjects_list.txt \
                     --sd $fastsurferdir \
                     --threads 4 --3T
```

### Flags
The `brun_fastsurfer.sh` script accepts almost all `run_fastsurfer.sh` flags (exceptions are `--t1` and `--sid`). In addition, it has [powerful parallelization options](../scripts/BATCH.md#parallelization-with-brun_fastsurfersh).

Example 5: Quick Segmentation
-----------------------------
For many applications you won't need the surfaces. You can run only the aparc+DKT segmentation (in 1 minute on a GPU) via

```bash
./run_fastsurfer.sh --t1 $datadir/subject1/t1-weighted.nii.gz \
                    --asegdkt_segfile $outputdir/subject1/aparc.DKTatlas+aseg.deep.mgz \
                    --conformed_name $outputdir/subject1/conformed.mgz \
                    --sd $HOME/my_fastsurfer_analysis \
                    --sid subject1 \
                    --threads 4 --seg_only --no_cereb --no_hypothal
```

This will produce the segmentation in a conformed space (just as FreeSurfer would do). It also writes the conformed image that fits the segmentation.
Conformed means that the image will be isotropic in LIA orientation.
It will furthermore output a brain mask (`mri/mask.mgz`), a simplified segmentation file (`mri/aseg.auto_noCCseg.mgz`), the biasfield corrected image (`mri/orig_nu.mgz`), and the volume statistics (without eTIV) based on the FastSurferVINN segmentation (without the corpus callosum) (`stats/aseg+DKT.stats`).

If you do not even need the biasfield corrected image and the volume statistics, you may add `--no_biasfield`. These steps especially benefit from larger assigned core counts `--threads 32`.

The above ```run_fastsurfer.sh``` commands can also be called from the Docker or Singularity images by passing the flags and adjusting input and output directories to the locations inside the containers (where you mapped them via the -v flag in Docker or -B in Singularity).

```bash
# Docker
docker run --gpus all \
           -v $HOME/my_mri_data:$HOME/my_mri_data \
           -v $HOME/my_fastsurfer_analysis:$HOME/my_fastsurfer_analysis \
           -v $HOME/my_freesurfer_license.txt:$HOME/my_freesurfer_license.txt \
           --rm --user $(id -u):$(id -g) deepmi/fastsurfer:latest \
              --t1 $HOME/my_mri_data/subject1/t1-weighted.nii.gz \
              --asegdkt_segfile $HOME/my_fastsurfer_analysis/subject1/aparc.DKTatlas+aseg.deep.mgz \
              --conformed_name $HOME/my_fastsurfer_analysis/subject1/conformed.mgz \
              --sd $HOME/my_fastsurfer_analysis \
              --sid subject1 \
              --threads 4 --seg_only --3T --no_cereb --no_hypothal
```

Example 6: Running FastSurfer on a SLURM cluster via Singularity
----------------------------------------------------------------
Starting with version 2.2, FastSurfer comes with a script that helps orchestrate FastSurfer optimally on a SLURM cluster: `srun_fastsurfer.sh`.

This script distributes GPU-heavy and CPU-heavy workloads to different SLURM partitions and manages intermediate files in a work directory for IO performance.

```bash
srun_fastsurfer.sh --partition seg=GPU_Partition \
                   --partition surf=CPU_Partition \
                   --sd $HOME/my_fastsurfer_analysis \
                   --data $HOME/my_mri_data \
                   --pattern */t1-weighted.nii.gz \
                   --remove_suffix /t1-weighted.nii.gz \
                   --singularity_image $HOME/images/fastsurfer-singularity.sif \
                   [...] # fastsurfer flags
```

This will create three dependent SLURM jobs, one to segment, one for surface reconstruction and one for cleanup (which moves the data from the work directory to the `$outputdir`).
There are many intricacies and options, so it is advised to use `--help`, `--debug` and `--dry` to inspect, what will be scheduled as well as run a test on a small subset. More control over subjects is available with `--subject_list`.

The `$HOME/my_mri_data` and the `$HOME/my_fastsurfer_analysis` directories need to be accessible from cluster nodes. Most IO is performed on a work directory (automatically generated from `$HPCWORK` environment variable: `$HPCWORK/fastsurfer-processing/$(date +%Y%m%d-%H%M%S)`). Alternatively, an empty directory can be manually defined via `--work`. On successful cleanup, this directory will be removed to `$HOME/my_fastsurfer_analysis` (defined via `--sd`).
