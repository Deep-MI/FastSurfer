Examples
========
Example 1: FastSurfer Apptainer (or Singularity)
------------------------------------------------
Apptainer (the open-source continuation of Singularity) is often available in HPC settings. Build the image as
described for [Linux](install/LINUX.md#apptainer-or-singularity). For the surface reconstruction, you need a
[FreeSurfer license](INSTALL.md#freesurfer-license).

To run FastSurfer on a subject with GPU access, execute:

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

The output will be written to `$HOME/my_fastsurfer_analysis/subjectX/`.
[Running FastSurfer in a container](CONTAINERS.md#using-apptainer-or-singularity) explains the
[Apptainer flags](CONTAINERS.md#apptainer-flags) and the
[FastSurfer flags](CONTAINERS.md#fastsurfer-flags-in-a-container). Without a GPU, use the
[CPU image](install/LINUX.md#cpu-only) and leave out `--nv`.

Example 2: FastSurfer Docker
----------------------------
Download the image as described for [Linux](install/LINUX.md#docker) or [Windows](install/WINDOWS.md); it includes
FreeSurfer, so you do not need to install it. For the surface reconstruction, you need a
[FreeSurfer license](INSTALL.md#freesurfer-license).

To run FastSurfer on a subject with GPU access, execute:

```bash
freesurfer_license=${freesurfer_license:-/path/to/your/freesurfer/license_file}
docker run --gpus all -v $HOME/my_mri_data:$HOME/my_mri_data \
    -v $HOME/my_fastsurfer_analysis:$HOME/my_fastsurfer_analysis \
    -v $freesurfer_license:$freesurfer_license \
    --rm --user $(id -u):$(id -g) \
    deepmi/fastsurfer:{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }} \
    --fs_license $freesurfer_license \
    --t1 $HOME/my_mri_data/subjectX/t1_weighted.nii.gz \
    --sid subjectX --sd $HOME/my_fastsurfer_analysis \
    --3T --threads 4
```

The output will be written to `$HOME/my_fastsurfer_analysis/subjectX/`.
[Running FastSurfer in a container](CONTAINERS.md#using-docker) explains the [Docker flags](CONTAINERS.md#docker-flags)
and the [FastSurfer flags](CONTAINERS.md#fastsurfer-flags-in-a-container). Without a GPU, use the
[CPU image](install/LINUX.md#cpu-only) and leave out `--gpus all`.

Example 3: Native FastSurfer on subjectX with parallel processing of hemis
--------------------------------------------------------------------------
For a native install you may want to make sure that you are on our stable branch, as the default dev branch is for development and could be broken at any time. For that you can directly clone the stable branch:

```bash
export FASTSURFER_HOME=${FASTSURFER_HOME:-/path/to/FastSurfer}
git clone --branch stable https://github.com/Deep-MI/FastSurfer.git \
    $FASTSURFER_HOME
```

More details (e.g. you need all dependencies in the right versions and also FreeSurfer locally) can be found in our [guide to installing from source](install/NATIVE.md).
Given you want to analyze data for subject which is stored on your computer under `$HOME/my_mri_data/subjectX/t1_weighted.nii.gz`, run the following command from the console (do not forget to source FreeSurfer!):

```bash
export FASTSURFER_HOME=${FASTSURFER_HOME:-/path/to/FastSurfer}

# Source FreeSurfer
export FREESURFER_HOME=${FREESURFER_HOME:-/path/to/freesurfer}
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Define data directory
data_dir=$HOME/my_mri_data
output_dir=$HOME/my_fastsurfer_analysis

# Run FastSurfer
$FASTSURFER_HOME/run_fastsurfer.sh --t1 $data_dir/subjectX/t1_weighted.nii.gz \
                    --sid subjectX --sd $output_dir \
                    --threads 4 --3T
```

The output will be stored in the `$output_dir` (including the `aparc.DKTatlas+aseg.deep.mgz` segmentation under `$output_dir/subjectX/mri` (default location)). For surfaces `--threads` is a total budget that the two hemispheres split and use at the same time, so `--threads 4` gives two threads per hemisphere. Without the flag, FastSurfer chooses the number itself (see `--threads` in [run_fastsurfer.sh](../scripts/RUN_FASTSURFER.md)). Pass `--threads 1` to run everything in a single thread, one hemisphere after the other, e.g. if you want to save resources on a compute cluster.


Example 4: FastSurfer on multiple subjects
------------------------------------------
In order to run FastSurfer on multiple cases, you may use the helper script `brun_fastsurfer.sh`. This script accepts multiple ways to define the subjects, for example a subjects_list file.
Prepare the subjects_list file as follows (one line subject per line; delimited by `\n`):
```text
<subject_id_1>=<t1_path_1>
<subject_id_2>=<t1_path_2>
<subject_id_3>=<t1_path_3>
...
<subject_id_10>=<t1_path_10>
```
Note, that all paths (`<t1_path_1>`, ...) are as if you passed them to the `run_fastsurfer.sh` script via `--t1 <t1_path>` so they may be with respect to the singularity or docker file system. Absolute paths are recommended.

The `brun_fastsurfer.sh` script can then be invoked in docker, singularity or on the native platform as follows:

### Docker
```bash
freesurfer_license=${freesurfer_license:-/path/to/your/freesurfer/license_file}
docker run --gpus all -v $HOME/my_mri_data:$HOME/my_mri_data \
    -v $HOME/my_fastsurfer_analysis:$HOME/my_fastsurfer_analysis \
    -v $freesurfer_license:$freesurfer_license \
    --entrypoint "/fastsurfer/brun_fastsurfer.sh" \
    --rm --user $(id -u):$(id -g) \
    deepmi/fastsurfer:{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }} \
    --fs_license $freesurfer_license \
    --sd $HOME/my_fastsurfer_analysis \
    --subjects_list $HOME/my_mri_data/subjects_list.txt \
    --3T \
    --threads 4
```
### Singularity
```bash
freesurfer_license=${freesurfer_license:-/path/to/your/freesurfer/license_file}
singularity exec --nv \
                 --no-mount home,cwd \
                 -B $HOME/my_mri_data \
                 -B $HOME/my_fastsurfer_analysis \
                 -B $freesurfer_license \
                 $HOME/my_singularity_images/fastsurfer-{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}.sif \
                 /fastsurfer/brun_fastsurfer.sh \
                 --fs_license $freesurfer_license \
                 --sd $HOME/my_fastsurfer_analysis \
                 --subjects_list $HOME/my_mri_data/subjects_list.txt \
                 --3T \
                 --threads 4
```
### Native
```bash
export FASTSURFER_HOME=${FASTSURFER_HOME:-/path/to/FastSurfer}
export FREESURFER_HOME=${FREESURFER_HOME:-/path/to/freesurfer}
source $FREESURFER_HOME/SetUpFreeSurfer.sh

data_dir=$HOME/my_mri_data
output_dir=$HOME/my_fastsurfer_analysis

# Run FastSurfer
$FASTSURFER_HOME/brun_fastsurfer.sh \
                     --subjects_list $data_dir/subjects_list.txt \
                     --sd $output_dir \
                     --threads 4 --3T
```

### Flags
The `brun_fastsurfer.sh` script accepts almost all `run_fastsurfer.sh` flags (exceptions are `--t1` and `--sid`). In addition, it has [powerful parallelization options](../scripts/BATCH.md#parallelization-with-brun_fastsurfersh).

Example 5: Quick Segmentation
-----------------------------
For many applications you won't need the surfaces. You can run only the aparc+DKT segmentation (in 1 minute on a GPU) via

```bash
export FASTSURFER_HOME=${FASTSURFER_HOME:-/path/to/FastSurfer}
$FASTSURFER_HOME/run_fastsurfer.sh \
    --t1 $HOME/my_mri_data/subjectX/t1_weighted.nii.gz \
    --asegdkt_segfile \
      $HOME/my_fastsurfer_analysis/subjectX/aparc.DKTatlas+aseg.deep.mgz \
    --conformed_name $HOME/my_fastsurfer_analysis/subjectX/conformed.mgz \
    --sd $HOME/my_fastsurfer_analysis \
    --sid subjectX \
    --threads 4 --seg_only --no_cereb --no_hypothal
```

This will produce the segmentation in a conformed space (just as FreeSurfer would do). It also writes the conformed image that fits the segmentation.
Conformed means that the image will be isotropic in LIA orientation.
It will furthermore output a brain mask (`mri/mask.mgz`), a simplified segmentation file (`mri/aseg.auto_noCCseg.mgz`), the biasfield corrected image (`mri/orig_nu.mgz`), and the volume statistics (without eTIV) based on the FastSurferVINN segmentation (without the corpus callosum) (`stats/aseg+DKT.stats`).

If you do not even need the biasfield corrected image and the volume statistics, you may add `--no_biasfield`. These steps especially benefit from larger assigned core counts `--threads 32`.

The above ```run_fastsurfer.sh``` commands can also be called from the Docker or Singularity images by passing the flags and adjusting input and output directories to the locations inside the containers (where you mapped them via the -v flag in Docker or -B in Singularity, here at the same paths as on your system).

```bash
# Docker
docker run --gpus all \
    -v $HOME/my_mri_data:$HOME/my_mri_data \
    -v $HOME/my_fastsurfer_analysis:$HOME/my_fastsurfer_analysis \
    --rm --user $(id -u):$(id -g) \
    deepmi/fastsurfer:{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }} \
      --t1 $HOME/my_mri_data/subjectX/t1_weighted.nii.gz \
      --asegdkt_segfile \
        $HOME/my_fastsurfer_analysis/subjectX/aparc.DKTatlas+aseg.deep.mgz \
      --conformed_name $HOME/my_fastsurfer_analysis/subjectX/conformed.mgz \
      --sd $HOME/my_fastsurfer_analysis \
      --sid subjectX \
      --threads 4 --seg_only --3T --no_cereb --no_hypothal
```

Example 6: Running FastSurfer on a SLURM cluster via Singularity
----------------------------------------------------------------
FastSurfer comes with a script that helps orchestrate FastSurfer optimally on a SLURM cluster: `srun_fastsurfer.sh`.

This script distributes GPU-heavy and CPU-heavy workloads to different SLURM partitions and manages intermediate files in a work directory for IO performance.

```bash
export FASTSURFER_HOME=${FASTSURFER_HOME:-/path/to/FastSurfer}
$FASTSURFER_HOME/srun_fastsurfer.sh --partition_seg GPU_Partition \
    --partition_surf CPU_Partition \
    --sd $HOME/my_fastsurfer_analysis \
    --data $HOME/my_mri_data \
    --pattern '*/t1_weighted.nii.gz' \
    --remove_suffix /t1_weighted.nii.gz \
    --singularity_image \
        $HOME/my_singularity_images/fastsurfer-{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}.sif \
    --3T # fastsurfer flags
```

This will create three dependent SLURM jobs, one to segment, one for surface reconstruction and one for cleanup (which moves the data from the work directory to `$HOME/my_fastsurfer_analysis`).
There are many intricacies and options, so it is advised to use `--help`, `--debug` and `--dry` to inspect, what will be scheduled as well as run a test on a small subset. More control over subjects is available with `--subjects_list`.

The `$HOME/my_mri_data` and the `$HOME/my_fastsurfer_analysis` directories need to be accessible from cluster nodes. Most IO is performed on a work directory (automatically generated from `$HPCWORK` environment variable: `$HPCWORK/fastsurfer-processing/$(date +%Y%m%d-%H%M%S)`). Alternatively, an empty directory can be manually defined via `--work`. On successful cleanup, this directory will be removed to `$HOME/my_fastsurfer_analysis` (defined via `--sd`).

## Example 7: Running FastSurfer with lesion inpainting using neurolit

When T1w images contain large lesions such as tumors, surgical cavities, or other abnormalities,
FastSurfer segmentation and surfaces can be affected by the altered anatomy. FastSurfer can be
wrapped with the Lesion Inpainting Tool (LIT) by providing `--lesion_mask <lesion_mask_path>`.

> **Note:** Review the LIT-modified outputs before using them for downstream analyses.

### Docker
```bash
freesurfer_license=${freesurfer_license:-/path/to/your/freesurfer/license_file}
docker run --gpus all -v $HOME/my_mri_data:$HOME/my_mri_data \
    -v $HOME/my_fastsurfer_analysis:$HOME/my_fastsurfer_analysis \
    -v $freesurfer_license:$freesurfer_license \
    --rm --user $(id -u):$(id -g) \
    deepmi/fastsurfer:{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }} \
    --fs_license $freesurfer_license \
    --t1 $HOME/my_mri_data/subjectX/t1_weighted.nii.gz \
    --lesion_mask $HOME/my_mri_data/subjectX/lesion_mask.nii.gz \
    --sid subjectX --sd $HOME/my_fastsurfer_analysis \
    --threads 4
```

### Native
```bash
export FASTSURFER_HOME=${FASTSURFER_HOME:-/path/to/FastSurfer}
export FREESURFER_HOME=${FREESURFER_HOME:-/path/to/freesurfer}
source $FREESURFER_HOME/SetUpFreeSurfer.sh
freesurfer_license=${freesurfer_license:-/path/to/your/freesurfer/license_file}

$FASTSURFER_HOME/run_fastsurfer.sh \
    --t1 $HOME/my_mri_data/subjectX/t1_weighted.nii.gz \
    --lesion_mask $HOME/my_mri_data/subjectX/lesion_mask.nii.gz \
    --sid subjectX --sd $HOME/my_fastsurfer_analysis \
    --fs_license $freesurfer_license \
    --threads 4
```

When using `--lesion_mask <lesion_mask_path>`, FastSurfer will:
1. Inpaint the lesion area using LIT.
2. Run the selected parts of the segmentation and surface pipeline on the inpainted image (note that it is not compatible with `--surf_only`).
3. Automatically map the lesion mask back into the final output files and regenerate the affected statistics.
4. Preserve the pre-lesion outputs as `.lit` or mapped backup files and write lesion reports plus `lesion_impact_summary.json` in the `stats` directory.
