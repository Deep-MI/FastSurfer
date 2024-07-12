# Overview - recon-surf

This directory contains all information needed to run the surface-generation and processing part of FastSurfer. Within 
approximately 1-1.5 h (depending on processing parallelization and image quality) this pipeline provides a fast FreeSurfer 
alternative for cortical surface reconstruction, mapping of cortical labels and traditional point-wise and ROI thickness analysis.
The basis for the reconstruction pipeline is the accurate anatomical whole brain segmentation following the DKTatlas 
such as the one provided by the FastSurferCNN or FastSurferVINN deep learning architectures.

The T1-weighted full head input image and the segmentation need to be equivalent in voxel size, dimension and orientation (LIA). 
With FastSurferCNN or VINN this is always ensured. If the image resolution is below 0.999, the surface pipeline
will be run in hires mode.  

Also note, that if a file exists at `$subjects_dir/$subject_id/mri/orig_nu.mgz`, this file will be used as the bias-field corrected image and the bias-field correction is skipped. 

# Usage
The *recon_surf* directory contains scripts to run the analysis. In addition, a working installation of __FreeSurfer__ (the supported version, usually the most recent) is needed for a native install (or use our Docker/Singularity images). 

The main script is called __recon-surf.sh__ which accepts certain arguments via the command line.
List them by running the following command:

```bash
./recon-surf.sh --help
```

## Required arguments
* `--sd`: Output directory \$SUBJECTS_DIR (equivalent to FreeSurfer setup --> $SUBJECTS_DIR/sid/mri; $SUBJECTS_DIR/sid/surf ... will be created).
* `--sid`: Subject ID for directory inside \$SUBJECTS_DIR to be created ($SUBJECTS_DIR/sid/...)

## Optional arguments
* `--t1`: T1 full head input (not bias corrected). This must be conformed (dimensions: same along each axis, voxel size: isotropic, LIA orientation, and data type UCHAR). Images can be conformed using FastSurferCNN's [conform.py](https://github.com/Deep-MI/FastSurfer/blob/stable/FastSurferCNN/data_loader/conform.py) script (usage example: python3 FastSurferCNN/data_loader/conform.py -i <T1_input> -o <conformed_T1_output>). If not passed we use the orig.mgz in the output subject mri directory if available. 
* `--asegdkt_segfile`: Global path with filename of segmentation (where and under which name to find it, must already exist). This must be conformed (dimensions: same along each axis, voxel size: isotropic, and LIA orientation). FastSurferCNN's segmentations are conformed by default. Please ensure that segmentations produced otherwise are also conformed and equivalent in dimension and voxel size to the --t1 image. Default location: $SUBJECTS_DIR/$sid/mri/aparc.DKTatlas+aseg.deep.mgz 
* `--3T`: for Talairach registration, use the 3T atlas instead of the 1.5T atlas (which is used if the flag is not provided). This gives better (more consistent with FreeSurfer) ICV estimates (eTIV) for 3T and better Talairach registration matrices, but has little impact on standard volume or surface stats.
* `--fstess`: Use mri_tesselate instead of marching cube (default) for surface creation
* `--fsqsphere`: Use FreeSurfer default instead of novel spectral spherical projection for qsphere
* `--fsaparc`: Use FS aparc segmentations in addition to DL prediction (slower in this case and usually the mapped ones from the DL prediction are fine)
* `--parallel`: Run both hemispheres in parallel
* `--threads`: Set openMP and ITK threads to <int>

## Other
* `--py`: Command for python, used in both pipelines. Default: python3.10
* `--no_surfreg`: Skip surface registration with FreeSurfer (if only stats are needed)
* `--fs_license`: Path to FreeSurfer license key file. Register at https://surfer.nmr.mgh.harvard.edu/registration.html for free to obtain it if you do not have FreeSurfer installed already

For more details see `--help`.

### Example 1: Surface module inside Docker

Docker can be used to simplify the installation (no FreeSurfer on system required). 
Given you already ran the segmentation pipeline, and want to just run the surface pipeline on top of it 
(i.e. on a different cluster), the following command can be used:
```bash
# 1. Pull the docker image (if it does not exist locally)
docker pull deepmi/fastsurfer:cpu-v?.?.?

# 2. Run command
docker run -v /home/user/my_fastsurfer_analysis:/output \
           -v /home/user/my_fs_license_dir:/fs_license \
           --rm --user $(id -u):$(id -g) deepmi/fastsurfer:cpu-v?.?.? \
           --fs_license /fs_license/license.txt \
           --sid subjectX --sd /output --3T --surf_only
```
Check [Dockerhub](https://hub.docker.com/r/deepmi/fastsurfer/tags) to find out the latest release version and replace the "?". 

Docker Flags: 
* The `-v` commands mount your output, and directory with the FreeSurfer license file into the Docker container. Inside the container these are visible under the name following the colon (in this case /output and /fs_license). 

This essentially calls the run_fastsurfer.sh script as entry point and starts only the surface module. It assumes that this case `subjectX` exists already and that the output files of the segmentation module are 
available in the subjectX/mri directory (e.g. `/home/user/my_fastsurfeer_analysis/subjectX/mri/aparc.DKTatlas+aseg.deep.mgz`, `mask.mgz`, `orig.mgz` etc.). The directory will then be populated with the FreeSurfer file structure, including surfaces, statistics and labels file (equivalent to a FreeSurfer recon-all run). It is possible to modify the entry point during the docker call and directly run recon-surf.sh, as we will demonstrate with the Singularity example next.

## Example 2: recon-surf inside Singularity
Singularity can be used instead of Docker to run the full pipeline or individual modules. In this example we change the entrypoint to `recon-surf.sh` instead of the standard
`run_fastsurfer.sh`. Usually it is recomended to just use the default, so this is for expert users who may want to try out specific flags that are not passed to the wrapper. 
Given you already ran the segmentation pipeline, and want to just run 
the surface pipeline on top of it (i.e. on a different cluster), the following command can be used:
```bash
# 1. Build the singularity image (if it does not exist)
singularity build fastsurfer-cpu-v?.?.?.sif docker://deepmi/fastsurfer:cpu-v?.?.?

# 2. Run command
singularity exec --no-home \
                 -B /home/user/my_fastsurfer_analysis:/output \
                 -B /home/user/my_fs_license_dir:/fs_license \
                  ./fastsurfer-cpu-?.?.?.sif \
                  /fastsurfer/recon_surf/recon-surf.sh \
                  --fs_license /fs_license/license.txt \
                  --sid subjectX --sd /output --3T \
                  --t1 <path_to>/subjectX/mri/orig.mgz \
                  --asegdkt_segfile <path_to>/subjectX/mri/aparc.DKTatlas+aseg.deep.mgz
```
Check [Dockerhub](https://hub.docker.com/r/deepmi/fastsurfer/tags) to find out the latest release version and replace the "?". 

### Singularity Flags: 
* The `-B` commands mount your output, and directory with the FreeSurfer license file into the Singularity container. Inside the container these are visible under the name following the colon (in this case /data, /output, and /fs_license). 

* The `--no-home` command disables the automatic mount of the users home directory (see [Best Practice](../Singularity/README.md#mounting-home))

The `--t1` and `--asegdkt_segfile` flags point to the already existing conformed T1 input and segmentation from the segmentation module. Also other files from that pipeline
will be reused (e.g. the `mask.mgz`, `orig_nu.mgz`). The subject directory will then be populated with the FreeSurfer file structure, including surfaces, statistics 
and labels file (equivalent to a FreeSurfer recon-all run). 

## Example 3: Native installation - recon-surf on a single subject (subjectX)

Given you want to analyze data for subjectX which is stored on your computer under `/home/user/my_mri_data/subjectX/orig.mgz`, 
run the following command from the console (do not forget to source FreeSurfer!):

```bash
# Source FreeSurfer
export FREESURFER_HOME=/path/to/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Define data directory
datadir=/home/user/my_mri_data
segdir=/home/user/my_segmentation_data
targetdir=/home/user/my_recon_surf_output  # equivalent to FreeSurfer's SUBJECTS_DIR

# Run recon-surf
./recon-surf.sh --sid subjectX \
                --sd $targetdir \
                --py python3.10 \
                --3T \
                --t1 <path_to>/subjectX/mri/orig.mgz \
                --asegdkt_segfile <path_to>/subjectX/mri/aparc.DKTatlas+aseg.deep.mgz
```

The `--t1` and `--asegdkt_segfile` flags point to the already existing conformed T1 input and segmentation from the segmentation module. Also other files from that pipeline
will be reused (e.g. the `mask.mgz`, `orig_nu.mgz`, i.e. under `/home/user/my_fastsurfeer_analysis/subjectX/mri/mask.mgz`). The `subjectX` directory will then be populated with the FreeSurfer file structure, including surfaces, statistics and labels file (equivalent to a FreeSurfer recon-all run). 
The script will generate a bias-field corrected image at `/home/user/my_fastsurfeer_analysis/subjectX/mri/orig_nu.mgz`, if this did not already exist.

### Example 4: recon-surf on multiple subjects

Most of the recon_surf functionality can also be achieved by running `run_fastsurfer.sh` with the `--surf_only` flag. This means we can also use the `brun_fastsurfer.sh` command with `--surf_only` to achieve similar results (see also [Example 4](../doc/overview/EXAMPLES.md#example-4-fastsurfer-on-multiple-subjects).

There are however some small differences to be aware of:
1. the path to and the filename of the t1 image in the subject_list file is optional.
2. you are not able to specify a custom, conformed t1 image via `--t1 <path>` (`run_fastsurfer.sh --seg_only` will always use `$subjects_dir/$subject_id/mri/orig.mgz`). 

Invoke the following command (make sure you have enough resources to run the given number of subjects in parallel or drop the `--parallel_subjects` flag to run them in series!):

```bash
singularity exec --no-home \
            -B /home/user/my_fastsurfer_analysis:/output \
            -B /home/user/subjects_lists/:/lists \
            -B /home/user/my_fs_license_dir:/fs_license \
            ./fastsurfer.sif \
            /fastsurfer/brun_fastsurfer.sh \
            --surf_only \
            --subjects_list /lists/subjects_list.txt \
            --parallel_subjects \
            --sd /output \
            --fs_license /fs_license/license.txt \
            --3T
```

A dedicated subfolder will be used for each subject within the target directory. 
As the `--t1` and ``--asegdkt_segfile` flags are not set, a subfolder within the target directory named after each subject (`$subject_id`) needs to exist and contain T1-weighted conformed image, 
mask and segmentations (as output by our FastSurfer segmentation networks, i.e. under `$subjects_dir/$subject_id/mri/orig.mgz`, `$subjects_dir/$subject_id/mri/mask.mgz`, and `$subjects_dir/$subject_id/mri/aparc.DKTatlas+aseg.deep.mgz`, respectively). 
The directory will then be populated with the FreeSurfer file structure, including surfaces, statistics and labels file (equivalent to a FreeSurfer recon-all run). The script will also generate a bias-field corrected image at `$subjects_dir/$subject_id/mri/orig_nu.mgz`, if this did not already exist.   

The logs of individual subject's processing can be found in `$subjects_dir/$subject_id/scripts`. On the standard out (e.g. the console), the output from multiple subjects will be interleaved, but each line prepended by the subject id. 


# Manual Edits

## Brainmask Edits

Currently, FastSurfer has only very limited functionality for manual edits due to missing entrypoints into the recon-surf script. Starting with FastSurfer v2 one frequently requested edit type (brainmask editing) is now possible, as the initial mask is created in the first segmentation stage. By running segmentation and surface processing in two steps, the mask can be edited in-between.

For a **Docker setup** one can:

1. Run segmentation only:
    ```bash
    docker run --gpus=all --rm --name $CONTAINER_NAME \
                          -v $PATH_TO_IMAGE_DIR:$IMAGE_DIR \
                          -v $PATH_TO_OUTPUT_DIR:$OUTPUT_DIR \
                          --user $UID:$GID deepmi/fastsurfer:gpu-v?.?.? \
                          --t1 $IMAGE_DIR/input.mgz \
                          --sd $OUTPUT_DIR \
                          --sid $SUBJECT_ID \
                          --seg_only
    ```
2. Modify the ```$PATH_TO_OUTPUT_DIR/$SUBJECT_ID/mri/mask.mgz``` file as required.
3. Run the following Docker command to run the surface processing pipeline (remove `--3T` if you are working with 1.5T data):
    ```bash
    docker run --rm --name $CONTAINER_NAME \
               -v $PATH_TO_OUTPUT_DIR:$OUTPUT_DIR \
               -v $PATH_TO_FS_LICENSE_DIR:$FS_LICENSE_DIR \
               --user $UID:$GID deepmi/fastsurfer:gpu-v?.?.? \
               --sid $SUBJECT_ID  \
               --sd $OUTPUT_DIR/$SUBJECT_ID \
               --surf_only --3T \
               --fs_license $FS_LICENSE_DIR/license_file
    ```

For a **local install** you can similarly:

1. Go to the FastSurfer directory, source FreeSurfer and run the segmentation step:
    ```bash
    cd $FASTSURFER_HOME
    source $FREESURFER_HOME/SetUpFreeSurfer.sh
    ./run_fastsurfer.sh --t1 $IMAGE_DIR/input.mgz --sd $OUTPUT_DIR --sid $SUBJECT_ID --seg_only
    ```
2. Modify the ```$OUTPUT_DIR/$SUBJECT_ID/mri/mask.mgz``` file.
3. Run the surface pipeline (remove `--3T` if you are working with 1.5T data):
    ```bash
    ./run_fastsurfer.sh --sd $OUTPUT_DIR --sid $SUBJECT_ID --fs_license $FS_LICENSE_DIR/license_file --surf_only --3T
    ```
