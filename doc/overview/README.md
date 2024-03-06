[![DOI](https://zenodo.org/badge/211859022.svg)](https://zenodo.org/badge/latestdoi/211859022)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deep-MI/FastSurfer/blob/stable/Tutorial/Tutorial_FastSurferCNN_QuickSeg.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deep-MI/FastSurfer/blob/stable/Tutorial/Complete_FastSurfer_Tutorial.ipynb)

# Overview

This README contains all information needed to run FastSurfer - a fast and accurate deep-learning based neuroimaging pipeline. FastSurfer provides a fully compatible [FreeSurfer](https://freesurfer.net/) alternative for volumetric analysis (within minutes) and surface-based thickness analysis (within only around 1h run time). 
FastSurfer is transitioning to sub-millimeter resolution support throughout the pipeline.

The FastSurfer pipeline consists of two main parts for segmentation and surface reconstruction.  

- the segmentation sub-pipeline (`seg`) employs advanced deep learning networks for fast, accurate segmentation and volumetric calculation of the whole brain and selected substructures.
- the surface sub-pipeline (`recon-surf`) reconstructs cortical surfaces, maps cortical labels and performs a traditional point-wise and ROI thickness analysis. 


# Segmentation Modules 
- approximately 5 minutes (GPU), `--seg_only` only runs this part
- Modules (all by default):
  1. `asegdkt:` FastSurferVINN for whole brain segmentation (deactivate with `--no_asegdkt`)
     - the core, outputs anatomical segmentation and cortical parcellation and statistics of 95 classes, mimics FreeSurfer’s DKTatlas.
     - requires a T1w image ([notes on input images](#requirements-to-input-images)), supports high-res (up to 0.7mm, experimental beyond that).
     - performs bias-field correction and calculates volume statistics corrected for partial volume effects (skipped if `--no_biasfield` is passed).
  2. `cereb:` CerebNet for cerebellum sub-segmentation (deactivate with `--no_cereb`)
     - requires `asegdkt_segfile`, outputs cerebellar sub-segmentation with detailed WM/GM delineation.
     - requires a T1w image ([notes on input images](#requirements-to-input-images)), which will be resampled to 1mm isotropic images (no native high-res support).
     - calculates volume statistics corrected for partial volume effects (skipped if `--no_biasfield` is passed).

# Surface reconstruction
- approximately 60-90 minutes, `--surf_only` runs only the surface part
- supports high-resolution images (up to 0.7mm, experimental beyond that)

# Requirements to input images
All pipeline parts and modules require good quality MRI images, preferably from a 3T MR scanner.
FastSurfer expects a similar image quality as FreeSurfer, so what works with FreeSurfer should also work with FastSurfer. 
Notwithstanding module-specific limitations, resolution should be between 1mm and 0.7mm isotropic (slice thickness should not exceed 1.5mm). Preferred sequence is Siemens MPRAGE or multi-echo MPRAGE. GE SPGR should also work. See `--vox_size` flag for high-res behaviour.s

![](../../images/teaser.png)


# Getting started
## Installation 
There are two ways to run FastSurfer (links are to installation instructions):

1. In a container ([Singularity](INSTALL.md#singularity) or [Docker](INSTALL.md#docker)) (OS: [Linux](INSTALL.md#linux), [Windows](INSTALL.md#windows), [MacOS on Intel](INSTALL.md#docker--currently-only-supported-for-intel-cpus-)),
2. As a [native install](INSTALL.md#native--ubuntu-2004-) (all OS for segmentation part). 

We recommended you use Singularity or Docker, especially if either is already installed on your system, because the [images we provide](https://hub.docker.com/r/deepmi/fastsurfer) conveniently include everything needed for FastSurfer, expect a  [FreeSurfer license file](https://surfer.nmr.mgh.harvard.edu/fswiki/License). We have detailed, per-OS Installation instructions in the [INSTALL.md file](INSTALL.md).

## Usage

All installation methods use the `run_fastsurfer.sh` call interface (replace `*fastsurfer-flags*` with [FastSurfer flags](#required-arguments)), which is the general starting point for FastSurfer. However, there are different ways to call this script depending on the installation, which we explain here:

1. For container installations, you need to define the hardware and mount the folders with the input (`/data`) and output data (`/output`):  
   (a) For __singularity__, the syntax is 
    ```
    singularity exec --nv \
                     --no-home \
                     -B /home/user/my_mri_data:/data \
                     -B /home/user/my_fastsurfer_analysis:/output \
                     -B /home/user/my_fs_license_dir:/fs_license \
                     ./fastsurfer-gpu.sif \
                     /fastsurfer/run_fastsurfer.sh 
                     *fastsurfer-flags*
   ```
   The `--nv` flag is needed to allow FastSurfer to run on the GPU (otherwise FastSurfer will run on the CPU).

   The `--no-home` flag tells singularity to not mount the home directory (see [Singularity README](Singularity/README.md#mounting-home) for more info).

   The `-B` flag is used to tell singularity, which folders FastSurfer can read and write to.
 
   See also __[Example 2](#example-2--fastSurfer-singularity)__ for a full singularity FastSurfer run command and [the Singularity README](Singularity/README.md#fastsurfer-singularity-image-usage) for details on more singularity flags.  

   (b) For __docker__, the syntax is
    ```
    docker run --gpus all \
               -v /home/user/my_mri_data:/data \
               -v /home/user/my_fastsurfer_analysis:/output \
               -v /home/user/my_fs_license_dir:/fs_license \
               --rm --user $(id -u):$(id -g) \
               deepmi/fastsurfer:latest \
               *fastsurfer-flags*
    ```
   The `--gpus` flag is needed to allow FastSurfer to run on the GPU (otherwise FastSurfer will run on the CPU).

   The `-v` flag is used to tell docker, which folders FastSurfer can read and write to.
 
   See also __[Example 1](#example-1--fastSurfer-docker)__ for a full FastSurfer run inside a Docker container and [the Docker README](Docker/README.md#docker-flags-) for more details on the docker flags including `--rm` and `--user`.

2. For a __native install__, you need to activate your FastSurfer environment (e.g. `conda activate fastsurfer_gpu`) and make sure you have added the FastSurfer path to your `PYTHONPATH` variable, e.g. `export PYTHONPATH=$(pwd)`. 

   You will then be able to run fastsurfer with `./run_fastsurfer.sh *fastsurfer-flags*`.

   See also [Example 3](#example-3--native-fastsurfer-on-subjectx--with-parallel-processing-of-hemis-) for an illustration of the commands to run the entire FastSurfer pipeline (FastSurferCNN + recon-surf) natively.

### FastSurfer Flags
Next, you will need to select the `*fastsurfer-flags*` and replace `*fastsurfer-flags*` with your options. Please see the Examples below for some example flags.

The `*fastsurfer-flags*` will usually include the subject directory (`--sd`; Note, this will be the mounted path - `/output` - for containers), the subject name/id (`--sid`) and the path to the input image (`--t1`). For example:

```bash
... --sd /output --sid test_subject --t1 /data/test_subject_t1.nii.gz --3T
```
Additionally, you can use `--seg_only` or `--surf_only` to only run a part of the pipeline or `--no_biasfield`, `--no_cereb` and `--no_asegdkt` to switch off some segmentation modules (see above).
Here, we have also added the `--3T` flag, which tells fastsurfer to register against the 3T atlas for ICV estimation (eTIV).

In the following, we give an overview of the most important options, but you can view a full list of options with 

```bash
./run_fastsurfer.sh --help
```


#### Required arguments
* `--sd`: Output directory \$SUBJECTS_DIR (equivalent to FreeSurfer setup --> $SUBJECTS_DIR/sid/mri; $SUBJECTS_DIR/sid/surf ... will be created).
* `--sid`: Subject ID for directory inside \$SUBJECTS_DIR to be created ($SUBJECTS_DIR/sid/...)
* `--t1`: T1 full head input (not bias corrected, global path). The network was trained with conformed images (UCHAR, 256x256x256, 1-0.7 mm voxels and standard slice orientation). These specifications are checked in the run_prediction.py script and the image is automatically conformed if it does not comply. Note, outputs will be in the conformed space (as in FreeSurfer). 

#### Required for docker when running surface module
* `--fs_license`: Path to FreeSurfer license key file (only needed for the surface module). Register (for free) at https://surfer.nmr.mgh.harvard.edu/registration.html to obtain it if you do not have FreeSurfer installed so far. Strictly necessary if you use Docker, optional for local install (your local FreeSurfer license will automatically be used). The license file is usually located in $FREESURFER_HOME/license.txt or $FREESURFER_HOME/.license . 

#### Segmentation pipeline arguments (optional)
* `--seg_only`: only run FastSurferCNN (generate segmentation, do not run the surface pipeline)
* `--seg_log`: Name and location for the log-file for the segmentation (FastSurferCNN). Default: $SUBJECTS_DIR/$sid/scripts/deep-seg.log
* `--viewagg_device`: Define where the view aggregation should be run on. Can be "auto" or a device (see --device). By default, the program checks if you have enough memory to run the view aggregation on the gpu. The total memory is considered for this decision. If this fails, or you actively overwrote the check with setting with "cpu" view agg is run on the cpu. Equivalently, if you pass a different device, view agg will be run on that device (no memory check will be done).
* `--device`: Select device for NN segmentation (_auto_, _cpu_, _cuda_, _cuda:<device_num>_, _mps_), where cuda means Nvidia GPU, you can select which one e.g. "cuda:1". Default: "auto", check GPU and then CPU. "mps" is for native MAC installs to use the Apple silicon (M-chip) GPU. 
* `--asegdkt_segfile`: Name of the segmentation file, which includes the aparc+DKTatlas-aseg segmentations. Requires an ABSOLUTE Path! Default location: \$SUBJECTS_DIR/\$sid/mri/aparc.DKTatlas+aseg.deep.mgz
* `--no_cereb`: Switch of the cerebellum sub-segmentation
* `--cereb_segfile`: Name of the cerebellum segmentation file. If not provided, this intermediate DL-based segmentation will not be stored, but only the merged segmentation will be stored (see --main_segfile <filename>). Requires an ABSOLUTE Path! Default location: \$SUBJECTS_DIR/\$sid/mri/cerebellum.CerebNet.nii.gz
* `--no_biasfield`: Deactivate the calculation of partial volume-corrected statistics.

#### Surface pipeline arguments (optional)
* `--surf_only`: only run the surface pipeline recon_surf. The segmentation created by FastSurferCNN must already exist in this case.
* `--3T`: for Talairach registration, use the 3T atlas instead of the 1.5T atlas (which is used if the flag is not provided). This gives better (more consistent with FreeSurfer) ICV estimates (eTIV) for 3T and better Talairach registration matrices, but has little impact on standard volume or surface stats.
* `--fstess`: Use mri_tesselate instead of marching cube (default) for surface creation
* `--fsqsphere`: Use FreeSurfer default instead of novel spectral spherical projection for qsphere
* `--fsaparc`: Use FS aparc segmentations in addition to DL prediction (slower in this case and usually the mapped ones from the DL prediction are fine)
* `--parallel`: Run both hemispheres in parallel
* `--no_fs_T1`: Do not generate T1.mgz (normalized nu.mgz included in standard FreeSurfer output) and create brainmask.mgz directly from norm.mgz instead. Saves 1:30 min.
* `--no_surfreg`: Skip the surface registration (`sphere.reg`), which is generated automatically by default. To safe time, use this flag to turn this off.

#### Other
* `--threads`: Target number of threads for all modules (segmentation, surface pipeline), `1` (default) forces FastSurfer to only really use one core. Note, that the default value may change in the future for better performance on multi-core architectures.
* `--vox_size`: Forces processing at a specific voxel size. If a number between 0.7 and 1 is specified (below is experimental) the T1w image is conformed to that isotropic voxel size and processed. 
  If "min" is specified (default), the voxel size is read from the size of the minimal voxel size (smallest per-direction voxel size) in the T1w image:
  If the minimal voxel size is bigger than 0.98mm, the image is conformed to 1mm isometric.
  If the minimal voxel size is smaller or equal to 0.98mm, the T1w image will be conformed to isometric voxels of that voxel size.
  The voxel size (whether set manually or derived) determines whether the surfaces are processed with highres options (below 1mm) or not.
* `--py`: Command for python, used in both pipelines. Default: python3.10
* `--conformed_name`: Name of the file in which the conformed input image will be saved. Default location: \$SUBJECTS_DIR/\$sid/mri/orig.mgz
* `--ignore_fs_version`: Switch on to avoid check for FreeSurfer version. Program will terminate if the supported version (see recon-surf.sh) is not sourced. Can be used for testing dev versions.
* `-h`, `--help`: Prints help text

### Example 1: FastSurfer Docker
After pulling one of our images from Dockerhub, you do not need to have a separate installation of FreeSurfer on your computer (it is already included in the Docker image). However, if you want to run ___more than just the segmentation CNN___, you need to register at the FreeSurfer website (https://surfer.nmr.mgh.harvard.edu/registration.html) to acquire a valid license for free. The directory containing the license needs to be mounted and passed to the script via the `--fs_license` flag. Basically for Docker (as for Singularity below) you are starting a container image (with the run command) and pass several parameters for that, e.g. if GPUs will be used and mounting (linking) the input and output directories to the inside of the container image. In the second half of that call you pass parameters to our run_fastsurfer.sh script that runs inside the container (e.g. where to find the FreeSurfer license file, and the input data and other flags). 

To run FastSurfer on a given subject using the provided GPU-Docker, execute the following command:

```bash
# 1. get the fastsurfer docker image (if it does not exist yet)
docker pull deepmi/fastsurfer 

# 2. Run command
docker run --gpus all -v /home/user/my_mri_data:/data \
                      -v /home/user/my_fastsurfer_analysis:/output \
                      -v /home/user/my_fs_license_dir:/fs_license \
                      --rm --user $(id -u):$(id -g) deepmi/fastsurfer:latest \
                      --fs_license /fs_license/license.txt \
                      --t1 /data/subjectX/t1-weighted.nii.gz \
                      --sid subjectX --sd /output \
                      --parallel --3T
```

Docker Flags:
* The `--gpus` flag is used to allow Docker to access GPU resources. With it, you can also specify how many GPUs to use. In the example above, _all_ will use all available GPUS. To use a single one (e.g. GPU 0), set `--gpus device=0`. To use multiple specific ones (e.g. GPU 0, 1 and 3), set `--gpus 'device=0,1,3'`.
* The `-v` commands mount your data, output, and directory with the FreeSurfer license file into the docker container. Inside the container these are visible under the name following the colon (in this case /data, /output, and /fs_license). 
* The `--rm` flag takes care of removing the container once the analysis finished. 
* The `--user $(id -u):$(id -g)` part automatically runs the container with your group- (id -g) and user-id (id -u). All generated files will then belong to the specified user. Without the flag, the docker container will be run as root which is discouraged.

FastSurfer Flags:
* The `--fs_license` points to your FreeSurfer license which needs to be available on your computer in the my_fs_license_dir that was mapped above. 
* The `--t1` points to the t1-weighted MRI image to analyse (full path, with mounted name inside docker: /home/user/my_mri_data => /data)
* The `--sid` is the subject ID name (output folder name)
* The `--sd` points to the output directory (its mounted name inside docker: /home/user/my_fastsurfer_analysis => /output)
* The `--parallel` activates processing left and right hemisphere in parallel
* The `--3T` changes the atlas for registration to the 3T atlas for better Talairach transforms and ICV estimates (eTIV)

Note, that the paths following `--fs_license`, `--t1`, and `--sd` are __inside__ the container, not global paths on your system, so they should point to the places where you mapped these paths above with the `-v` arguments (part after colon). 

A directory with the name as specified in `--sid` (here subjectX) will be created in the output directory if it does not exist. So in this example output will be written to /home/user/my_fastsurfer_analysis/subjectX/ . Make sure the output directory is empty, to avoid overwriting existing files. 

If you do not have a GPU, you can also run our CPU-Docker by dropping the `--gpus all` flag and specifying `--device cpu` at the end as a FastSurfer flag. See [Docker/README.md](Docker/README.md) for more details.

### Example 2: FastSurfer Singularity
After building the Singularity image (see below or instructions in ./Singularity/README.md), you also need to register at the FreeSurfer website (https://surfer.nmr.mgh.harvard.edu/registration.html) to acquire a valid license (for free) - same as when using Docker. This license needs to be passed to the script via the `--fs_license` flag. This is not necessary if you want to run the segmentation only.

To run FastSurfer on a given subject using the Singularity image with GPU access, execute the following commands from a directory where you want to store singularity images. This will create a singularity image from our Dockerhub image and execute it:

```bash
# 1. Build the singularity image (if it does not exist)
singularity build fastsurfer-gpu.sif docker://deepmi/fastsurfer

# 2. Run command
singularity exec --nv \
                 --no-home \
                 -B /home/user/my_mri_data:/data \
                 -B /home/user/my_fastsurfer_analysis:/output \
                 -B /home/user/my_fs_license_dir:/fs_license \
                 ./fastsurfer-gpu.sif \
                 /fastsurfer/run_fastsurfer.sh \
                 --fs_license /fs_license/license.txt \
                 --t1 /data/subjectX/t1-weighted.nii.gz \
                 --sid subjectX --sd /output \
                 --parallel --3T
```

#### Singularity Flags
* The `--nv` flag is used to access GPU resources. 
* The `--no-home` flag stops mounting your home directory into singularity.
* The `-B` commands mount your data, output, and directory with the FreeSurfer license file into the Singularity container. Inside the container these are visible under the name following the colon (in this case /data, /output, and /fs_license). 

#### FastSurfer Flags
* The `--fs_license` points to your FreeSurfer license which needs to be available on your computer in the my_fs_license_dir that was mapped above. 
* The `--t1` points to the t1-weighted MRI image to analyse (full path, with mounted name inside docker: /home/user/my_mri_data => /data)
* The `--sid` is the subject ID name (output folder name)
* The `--sd` points to the output directory (its mounted name inside docker: /home/user/my_fastsurfer_analysis => /output)
* The `--parallel` activates processing left and right hemisphere in parallel
* The `--3T` changes the atlas for registration to the 3T atlas for better Talairach transforms and ICV estimates (eTIV)

Note, that the paths following `--fs_license`, `--t1`, and `--sd` are __inside__ the container, not global paths on your system, so they should point to the places where you mapped these paths above with the `-v` arguments (part after colon).

A directory with the name as specified in `--sid` (here subjectX) will be created in the output directory. So in this example output will be written to /home/user/my_fastsurfer_analysis/subjectX/ . Make sure the output directory is empty, to avoid overwriting existing files. 

You can run the Singularity equivalent of CPU-Docker by building a Singularity image from the CPU-Docker image and excluding the `--nv` argument in your Singularity exec command. Also append `--device cpu` as a FastSurfer flag.


### Example 3: Native FastSurfer on subjectX (with parallel processing of hemis)

For a native install you may want to make sure that you are on our stable branch, as the default dev branch is for development and could be broken at any time. For that you can directly clone the stable branch:

```bash
git clone --branch stable https://github.com/Deep-MI/FastSurfer.git
```

More details (e.g. you need all dependencies in the right versions and also FreeSurfer locally) can be found in our [INSTALL.md file](INSTALL.md).
Given you want to analyze data for subject which is stored on your computer under /home/user/my_mri_data/subjectX/t1-weighted.nii.gz, run the following command from the console (do not forget to source FreeSurfer!):

```bash
# Source FreeSurfer
export FREESURFER_HOME=/path/to/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Define data directory
datadir=/home/user/my_mri_data
fastsurferdir=/home/user/my_fastsurfer_analysis

# Run FastSurfer
./run_fastsurfer.sh --t1 $datadir/subjectX/t1-weighted-nii.gz \
                    --sid subjectX --sd $fastsurferdir \
                    --parallel --threads 4 --3T
```

The output will be stored in the $fastsurferdir (including the aparc.DKTatlas+aseg.deep.mgz segmentation under $fastsurferdir/subjectX/mri (default location)). Processing of the hemispheres will be run in parallel (--parallel flag) to significantly speed-up surface creation. Omit this flag to run the processing sequentially, e.g. if you want to save resources on a compute cluster.


### Example 4: FastSurfer on multiple subjects

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

#### Docker
```bash
docker run --gpus all -v /home/user/my_mri_data:/data \
                      -v /home/user/my_fastsurfer_analysis:/output \
                      -v /home/user/my_fs_license_dir:/fs_license \
                      --entrypoint "/fastsurfer/brun_fastsurfer.sh" \
                      --rm --user $(id -u):$(id -g) deepmi/fastsurfer:latest \
                      --fs_license /fs_license/license.txt \
                      --sd /output --subject_list /data/subjects_list.txt \
                      --parallel --3T
```
#### Singularity
```bash
singularity exec --nv \
                 --no-home \
                 -B /home/user/my_mri_data:/data \
                 -B /home/user/my_fastsurfer_analysis:/output \
                 -B /home/user/my_fs_license_dir:/fs_license \
                 ./fastsurfer-gpu.sif \
                 /fastsurfer/brun_fastsurfer.sh \
                 --fs_license /fs_license/license.txt \
                 --sd /output \
                 --subject_list /data/subjects_list.txt \
                 --parallel --3T
```
#### Native
```bash
export FREESURFER_HOME=/path/to/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

cd /home/user/FastSurfer
datadir=/home/user/my_mri_data
fastsurferdir=/home/user/my_fastsurfer_analysis

# Run FastSurfer
./brun_fastsurfer.sh --subject_list $datadir/subjects_list.txt \
                     --sd $fastsurferdir \
                     --parallel --threads 4 --3T
```

#### Flags
The `brun_fastsurfer.sh` script accepts almost all `run_fastsurfer.sh` flags (exceptions are `--t1` and `--sid`). In addition, 
* the `--parallel_subjects` runs all subjects in parallel (experimental, parameter may change in future releases). This is particularly useful for surfaces computation `--surf_only`.
* to run segmentation in series, but surfaces in parallel, you may use `--parallel_subjects surf`.
* these options are in contrast (and in addition) to `--parallel`, which just parallelizes the hemispheres of one case.

### Example 5: Quick Segmentation

For many applications you won't need the surfaces. You can run only the aparc+DKT segmentation (in 1 minute on a GPU) via

```bash
./run_fastsurfer.sh --t1 $datadir/subject1/t1-weighted.nii.gz \
                    --asegdkt_segfile $outputdir/subject1/aparc.DKTatlas+aseg.deep.mgz \
                    --conformed_name $outputdir/subject1/conformed.mgz \
                    --threads 4 --seg_only --no_cereb
```

This will produce the segmentation in a conformed space (just as FreeSurfer would do). It also writes the conformed image that fits the segmentation.
Conformed means that the image will be isotropic in LIA orientation. 
It will furthermore output a brain mask (`mri/mask.mgz`), a simplified segmentation file (`mri/aseg.auto_noCCseg.mgz`), the biasfield corrected image (`mri/orig_nu.mgz`), and the volume statistics (without eTIV) based on the FastSurferVINN segmentation (without the corpus callosum) (`stats/aseg+DKT.stats`).

If you do not even need the biasfield corrected image and the volume statistics, you may add `--no_biasfield`. These steps especially benefit from larger assigned core counts `--threads 32`.

The above ```run_fastsurfer.sh``` commands can also be called from the Docker or Singularity images by passing the flags and adjusting input and output directories to the locations inside the containers (where you mapped them via the -v flag in Docker or -B in Singularity). 

```bash
# Docker
docker run --gpus all -v $datadir:/data \
                      -v $outputdir:/output \
                      --rm --user $(id -u):$(id -g) deepmi/fastsurfer:latest \
                      --t1 /data/subject1/t1-weighted.nii.gz \
                      --asegdkt_segfile /output/subject1/aparc.DKTatlas+aseg.deep.mgz \
                      --conformed_name $outputdir/subject1/conformed.mgz \
                      --threads 4 --seg_only --3T
```

### Example 6: Running FastSurfer on a SLURM cluster via Singularity

Starting with version 2.2, FastSurfer comes with a script that helps orchestrate FastSurfer optimally on a SLURM cluster: `srun_fastsurfer.sh`.

This script distributes GPU-heavy and CPU-heavy workloads to different SLURM partitions and manages intermediate files in a work directory for IO performance.

```bash
srun_fastsurfer.sh --partition seg=GPU_Partition \
                   --partition surf=CPU_Partition \
                   --sd $outputdir \
                   --data $datadir \
                   --singularity_image $HOME/images/fastsurfer-singularity.sif \
                   [...] # fastsurfer flags
```

This will create three dependent SLURM jobs, one to segment, one for surface reconstruction and one for cleanup (which moves the data from the work directory to the `$outputdir`).
There are many intricacies and options, so it is advised to use `--help`, `--debug` and `--dry` to inspect, what will be scheduled as well as run a test on a small subset. More control over subjects is available with `--subject_list`s.

The `$outputdir` and the `$datadir` need to be accessible from cluster nodes. Most IO is performed on a work directory (automatically generated from `$HPCWORK` environment variable: `$HPCWORK/fastsurfer-processing/$(date +%Y%m%d-%H%M%S)`). Alternatively, an empty directory can be manually defined via `--work`. On successful cleanup, this directory will be removed.

## Output files


### Segmentation module

The segmentation module outputs the files shown in the table below. The two primary output files are the `aparc.DKTatlas+aseg.deep.mgz` file, which contains the FastSurfer segmentation of cortical and subcortical structures based on the DKT atlas, and the `aseg+DKT.stats` file, which contains summary statistics for these structures. Note, that the surface model (downstream) corrects these segmentations along the cortex with the created surfaces. So if the surface model is used, it is recommended to use the updated segmentations and stats (see below). 

| directory   | filename                      | module    | description |
|:------------|-------------------------------|-----------|-------------|
| mri         | aparc.DKTatlas+aseg.deep.mgz  | asegdkt   | cortical and subcortical segmentation|
| mri         | aseg.auto_noCCseg.mgz         | asegdkt   | simplified subcortical segmentation without corpus callosum labels|
| mri         | mask.mgz                      | asegdkt   | brainmask|
| mri         | orig.mgz                      | asegdkt   | conformed image|
| mri         | orig_nu.mgz                   | asegdkt   | biasfield-corrected image|
| mri/orig    | 001.mgz                       | asegdkt   | original image|
| scripts     | deep-seg.log                  | asegdkt   | logfile|
| stats       | aseg+DKT.stats                | asegdkt   | table of cortical and subcortical segmentation statistics|

### Cerebnet module

The cerebellum module outputs the files in the table shown below. Unless switched off by the `--no_cereb` argument, this module is automatically run whenever the segmentation module is run. It adds two files, an image with the sub-segmentation of the cerebellum and a text file with summary statistics.


| directory   | filename                      | module    | description |
|:------------|-------------------------------|-----------|-------------|
| mri         | cerebellum.CerebNet.nii.gz    | cerebnet  | cerebellum sub-segmentation|
| stats       | cerebellum.CerebNet.stats     | cerebnet  | table of cerebellum segmentation statistics|


### Surface module

The surface module is run unless switched off by the `--seg_only` argument. It outputs a large number of files, which generally correspond to the FreeSurfer nomenclature and definition. A selection of important output files is shown in the table below, for the other files, we refer to the [FreeSurfer documentation](https://surfer.nmr.mgh.harvard.edu/fswiki). In general, the "mri" directory contains images, including segmentations, the "surf" folder contains surface files (geometries and vertex-wise overlay data), the "label" folder contains cortical parcellation labels, and the "stats" folder contains tabular summary statistics. Many files are available for the left ("lh") and right ("rh") hemisphere of the brain. Symbolic links are created to map FastSurfer files to their FreeSurfer equivalents, which may need to be present for further processing (e.g., with FreeSurfer downstream modules). 

After running this module, some of the initial segmentations and corresponding volume estimates are fine-tuned (e.g., surface-based partial volume correction, addition of corpus callosum labels). Specifically, this concerns the `aseg.mgz `, `aparc.DKTatlas+aseg.mapped.mgz`, `aparc.DKTatlas+aseg.deep.withCC.mgz`, which were originally created by the segmentation module or have earlier versions resulting from that module.

The primary output files are pial, white, and inflated surface files, the thickness overlay files, and the cortical parcellation (annotation) files. The preferred way of assessing this output is the [FreeView](https://surfer.nmr.mgh.harvard.edu/fswiki/FreeviewGuide) software. Summary statistics for volume and thickness estimates per anatomical structure are reported in the stats files, in particular the `aseg.stats`, and the left and right `aparc.DKTatlas.mapped.stats` files. 

| directory   | filename                      | module    | description |
|:------------|-------------------------------|-----------|-------------|
| mri         | aparc.DKTatlas+aseg.deep.withCC.mgz| surface | cortical and subcortical segmentation incl. corpus callosum after running the surface module|
| mri         | aparc.DKTatlas+aseg.mapped.mgz| surface      | cortical and subcortical segmentation after running the surface module|
| mri         | aparc.DKTatlas+aseg.mgz       | surface      | symlink to aparc.DKTatlas+aseg.mapped.mgz|
| mri         | aparc+aseg.mgz                | surface      | symlink to aparc.DKTatlas+aseg.mapped.mgz|
| mri         | aseg.mgz                      | surface      | subcortical segmentation after running the surface module|
| mri         | wmparc.DKTatlas.mapped.mgz    | surface      | white matter parcellation|
| mri         | wmparc.mgz                    | surface      | symlink to wmparc.DKTatlas.mapped.mgz|
| surf        | lh.area, rh.area              | surface      | surface area overlay file|
| surf        | lh.curv, rh.curv              | surface      | curvature overlay file|
| surf        | lh.inflated, rh.inflated      | surface      | inflated cortical surface|
| surf        | lh.pial, rh.pial              | surface      | pial surface|
| surf        | lh.thickness, rh.thickness    | surface      | cortical thickness overlay file|
| surf        | lh.volume, rh.volume          | surface      | gray matter volume overlay file|
| surf        | lh.white, rh.white            | surface      | white matter surface|
| label       | lh.aparc.DKTatlas.annot, rh.aparc.DKTatlas.annot| surface      | symlink to lh.aparc.DKTatlas.mapped.annot|
| label       | lh.aparc.DKTatlas.mapped.annot, rh.aparc.DKTatlas.mapped.annot| surface      | annotation file for cortical parcellations, mapped from ASEGDKT segmentation to the surface|
| stats       | aseg.stats                    | surface      | table of cortical and subcortical segmentation statistics after running the surface module|
| stats       | lh.aparc.DKTatlas.mapped.stats, rh.aparc.DKTatlas.mapped.stats| surface      | table of cortical parcellation statistics, mapped from ASEGDKT segmentation to the surface|
| stats       | lh.curv.stats, rh.curv.stats  | surface      | table of curvature statistics|
| stats       | wmparc.DKTatlas.mapped.stats  | surface      | table of white matter segmentation statistics|
| scripts     | recon-all.log                 | surface      | logfile|


## System Requirements

Recommendation: At least 8 GB system memory and 8 GB NVIDIA graphics memory ``--viewagg_device gpu``  

Minimum: 7 GB system memory and 2 GB graphics memory ``--viewagg_device cpu --vox_size 1``

Minimum CPU-only: 8 GB system memory (much slower, not recommended) ``--device cpu --vox_size 1`` 

### Minimum Requirements:

|       | --viewagg_device | Min GPU (in GB) | Min CPU (in GB) |
|:------|------------------|----------------:|----------------:|
| 1mm   | gpu              |               5 |               5 |
| 1mm   | cpu              |               2 |               7 |
| 0.8mm | gpu              |               8 |               6 |
| 0.8mm | cpu              |               3 |               9 |
| 0.7mm | gpu              |               8 |               6 |
| 0.7mm | cpu              |               3 |               9 |

## Expert usage
Individual modules and the surface pipeline can be run independently of the full pipeline script documented in this README. 
This is documented in READMEs in subfolders, for example: [whole brain segmentation only with FastSurferVINN](FastSurferCNN/README.md), [cerebellum sub-segmentation (in progress)](CerebNet/README.md) and [surface pipeline only (recon-surf)](recon_surf/README.md).

Specifically, the segmentation modules feature options for optimized parallelization of batch processing.

## FreeSurfer Downstream Modules

FreeSurfer provides several Add-on modules for downstream processing, such as subfield segmentation ( [hippocampus/amygdala](https://surfer.nmr.mgh.harvard.edu/fswiki/HippocampalSubfieldsAndNucleiOfAmygdala), [brainstrem](https://surfer.nmr.mgh.harvard.edu/fswiki/BrainstemSubstructures), [thalamus](https://freesurfer.net/fswiki/ThalamicNuclei) and [hypothalamus](https://surfer.nmr.mgh.harvard.edu/fswiki/HypothalamicSubunits) ) as well as [TRACULA](https://surfer.nmr.mgh.harvard.edu/fswiki/Tracula). We now provide symlinks to the required files, as FastSurfer creates them with a different name (e.g. using "mapped" or "DKT" to make clear that these file are from our segmentation using the DKT Atlas protocol, and mapped to the surface). Most subfield segmentations require `wmparc.mgz` and work very well with FastSurfer,  so feel free to run those pipelines after FastSurfer. TRACULA requires `aparc+aseg.mgz` which we now link, but have not tested if it works, given that [DKT-atlas](https://mindboggle.readthedocs.io/en/latest/labels.html) merged a few labels. You should source FreeSurfer 7.3.2 to run these modules. 

## Intended Use

This software can be used to compute statistics from an MR image for research purposes. Estimates can be used to aggregate population data, compare groups etc. The data should not be used for clinical decision support in individual cases and, therefore, does not benefit the individual patient. Be aware that for a single image, produced results may be unreliable (e.g. due to head motion, imaging artefacts, processing errors etc). We always recommend to perform visual quality checks on your data, as also your MR-sequence may differ from the ones that we tested. No contributor shall be liable to any damages, see also our software [LICENSE](LICENSE). 

## References

If you use this for research publications, please cite:

_Henschel L, Conjeti S, Estrada S, Diers K, Fischl B, Reuter M, FastSurfer - A fast and accurate deep learning based neuroimaging pipeline, NeuroImage 219 (2020), 117012. https://doi.org/10.1016/j.neuroimage.2020.117012_

_Henschel L*, Kuegler D*, Reuter M. (*co-first). FastSurferVINN: Building Resolution-Independence into Deep Learning Segmentation Methods - A Solution for HighRes Brain MRI. NeuroImage 251 (2022), 118933. http://dx.doi.org/10.1016/j.neuroimage.2022.118933_

_Faber J*, Kuegler D*, Bahrami E*, et al. (*co-first). CerebNet: A fast and reliable deep-learning pipeline for detailed cerebellum sub-segmentation. NeuroImage 264 (2022), 119703. https://doi.org/10.1016/j.neuroimage.2022.119703_

Stay tuned for updates and follow us on Twitter: https://twitter.com/deepmilab

## Acknowledgements

This project is partially funded by:
- [Chan Zuckerberg Initiative](https://chanzuckerberg.com/eoss/proposals/fastsurfer-ai-based-neuroimage-analysis-package/)
- [German Federal Ministry of Education and Research](https://www.gesundheitsforschung-bmbf.de/de/deepni-innovative-deep-learning-methoden-fur-die-rechnergestutzte-neuro-bildgebung-10897.php)

The recon-surf pipeline is largely based on FreeSurfer 
https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferMethodsCitation

