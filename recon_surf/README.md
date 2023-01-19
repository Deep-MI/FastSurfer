# Overview - recon-surf

This directory contains all information needed to run the surface-generation and processing part of FastSurfer. Within 
approximately 1-1.5 h (depending on processing parallelization and image quality) this pipeline provides a fast FreeSurfer 
alternative for cortical surface reconstruction, mapping of cortical labels and traditional point-wise and ROI thickness analysis.
The basis for the reconstruction pipeline is the accurate anatomical whole brain segmentation following the DKTatlas 
such as the one provided by the FastSurferCNN or FastSurferVINN deep learning architectures.

The T1-weighted full head input image and the segmentation need to be equivalent in voxel size, dimension and orientation (LIA). 
With FastSurferCNN or VINN this is always ensured. If the image resolution is below 0.999, the surface pipeline
will be run in hires mode.  

# Usage
The *recon_surf* directory contains scripts to run the analysis. In addition, a working installation of __FreeSurfer__ (v7.3.2) is needed for a native install (or use our Docker/Singularity images). 

The main script is called __recon-surf.sh__ which accepts certain arguments via the command line.
List them by running the following command:

```bash
./recon-surf.sh --help
```

### Required arguments
* `--sd`: Output directory \$SUBJECTS_DIR (equivalent to FreeSurfer setup --> $SUBJECTS_DIR/sid/mri; $SUBJECTS_DIR/sid/surf ... will be created).
* `--sid`: Subject ID for directory inside \$SUBJECTS_DIR to be created ($SUBJECTS_DIR/sid/...)

### Optional arguments
* `--t1`: T1 full head input (not bias corrected). This must be conformed (dimensions: same along each axis, voxel size: isotropic, LIA orientation, and data type UCHAR). Images can be conformed using FastSurferCNN's [conform.py](https://github.com/Deep-MI/FastSurfer/blob/stable/FastSurferCNN/data_loader/conform.py) script (usage example: python3 FastSurferCNN/data_loader/conform.py -i <T1_input> -o <conformed_T1_output>). If not passed we use the orig.mgz in the output subject mri directory if available. 
* `--aparc_aseg_segfile`: Global path with filename of segmentation (where and under which name to find it, must already exist). This must be conformed (dimensions: same along each axis, voxel size: isotropic, and LIA orientation). FastSurferCNN's segmentations are conformed by default. Please ensure that segmentations produced otherwise are also conformed and equivalent in dimension and voxel size to the --t1 image. Default location: $SUBJECTS_DIR/$sid/mri/aparc.DKTatlas+aseg.deep.mgz 
* `--vol_segstats`: Additionally return volume-based aparc.DKTatlas+aseg statistics for DL-based segmentation (does not require surfaces and stops after cc-incorporation).
* `--fstess`: Use mri_tesselate instead of marching cube (default) for surface creation
* `--fsqsphere`: Use FreeSurfer default instead of novel spectral spherical projection for qsphere
* `--fsaparc`: Use FS aparc segmentations in addition to DL prediction (slower in this case and usually the mapped ones from the DL prediction are fine)
* `--parallel`: Run both hemispheres in parallel
* `--threads`: Set openMP and ITK threads to <int>

### Other
* `--py`: Command for python, used in both pipelines. Default: python3.8
* `--no_surfreg`: Skip surface registration with FreeSurfer (if only stats are needed)
* `--fs_license`: Path to FreeSurfer license key file. Register at https://surfer.nmr.mgh.harvard.edu/registration.html for free to obtain it if you do not have FreeSurfer installed already

For more details see `--help`.

### Example 1: recon-surf inside Docker

Docker can be again be used to simplify the installation (no FreeSurfer on system required). 
Given you already ran the segmentation pipeline, and want to just run the surface pipeline on top of it 
(i.e. on a different cluster), the following command can be used:
```bash
# 1. Build the singularity image (if it does not exist)
docker pull deepmi/fastsurfer:surfonly-cpu-v2.0.0

# 2. Run command
docker run -v /home/user/my_fastsurfer_analysis:/output \
           -v /home/user/my_fs_license_dir:/fs_license \
           --rm --user $(id -u):$(id -g) deepmi/fastsurfer:surfonly-cpu-v2.0.0 \
           --fs_license /fs_license/license.txt \
           --sid subjectX --sd /output 
```

Docker Flags: 
* The `-v` commands mount your output, and directory with the FreeSurfer license file into the Docker container. Inside the container these are visible under the name following the colon (in this case /output and /fs_license). 

As the --t1 and --aparc_aseg_segfile flag are not set, a subfolder within the target directory named after the subject (here: subjectX) needs to exist and contain t1-weighted conformed image, 
mask and segmentations (as output by our FastSurfer segmentation networks, i.e. under /home/user/my_fastsurfeer_analysis/subjectX/mri/aparc.DKTatlas+aseg.deep.mgz, mask.mgz, and orig.mgz)).  The directory will then be populated with the FreeSurfer file structure, including surfaces, statistics 
and labels file (equivalent to a FreeSurfer recon-all run). 


### Example 2: recon-surf inside Singularity
Singularity can be used as for the full pipeline. Given you already ran the segmentation pipeline, and want to just run 
the surface pipeline on top of it (i.e. on a different cluster), the following command can be used:
```bash
# 1. Build the singularity image (if it does not exist)
singularity build fastsurfer-reconsurf.sif docker://deepmi/fastsurfer:surfonly-cpu-v2.0.0

# 2. Run command
singularity exec -B /home/user/my_fastsurfer_analysis:/output \
                 -B /home/user/my_fs_license_dir:/fs_license \
                  ./fastsurfer-reconsurf.sif \
                  /fastsurfer/recon_surf/recon-surf.sh \
                  --fs_license /fs_license/license.txt \
                  --sid subjectX --sd /output 
```

Singularity Flags: 
* The `-B` commands mount your output, and directory with the FreeSurfer license file into the Singularity container. Inside the container these are visible under the name following the colon (in this case /data, /output, and /fs_license). 

As the --t1 and --aparc_aseg_segfile flag are not set, a subfolder within the target directory named after the subject (here: subjectX) needs to exist and contain t1-weighted conformed image, 
mask and segmentations (as output by our FastSurfer segmentation networks, i.e. under /home/user/my_fastsurfeer_analysis/subjectX/mri/aparc.DKTatlas+aseg.deep.mgz, mask.mgz, and orig.mgz)).  The directory will then be populated with the FreeSurfer file structure, including surfaces, statistics 
and labels file (equivalent to a FreeSurfer recon-all run). 

### Example 3: Native installation - recon-surf on a single subject (subjectX)

Given you want to analyze data for subjectX which is stored on your computer under /home/user/my_mri_data/subjectX/orig.mgz, 
run the following command from the console (do not forget to source FreeSurfer!):

```bash
# Source FreeSurfer
export FREESURFER_HOME=/path/to/freesurfer/fs732
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Define data directory
datadir=/home/user/my_mri_data
segdir=/home/user/my_segmentation_data
targetdir=/home/user/my_recon_surf_output  # equivalent to FreeSurfer's SUBJECT_DIR

# Run recon-surf
./recon-surf.sh --sid subjectX \
                --sd $targetdir \
                --py python3.8

```

As the --t1 and --aparc_aseg_segfile flag are not set, a subfolder within the target directory named after the subject (here: subjectX) needs to exist and contain t1-weighted conformed image, 
mask and segmentations (as output by our FastSurfer segmentation networks, i.e. under /home/user/my_fastsurfeer_analysis/subjectX/mri/aparc.DKTatlas+aseg.deep.mgz, mask.mgz, and orig.mgz)).  The directory will then be populated with the FreeSurfer file structure, including surfaces, statistics 
and labels file (equivalent to a FreeSurfer recon-all run). 

### Example 4: Native installation - recon-surf on multiple subjects (using nohup)

In order to run recon-surf on a number of cases which are stored in the same directory, prepare a subjects_list.txt file listing the names line per line:
subject1\n
subject2\n
subject3\n
...
subject10\n

And invoke the following command (make sure you have enough ressources to run the given number of subjects in parallel!):

```bash
# Source FreeSurfer
export FREESURFER_HOME=/path/to/freesurfer/fs732
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Define data directory
datadir=/home/user/my_mri_data
segdir=/home/user/my_segmentation_data
targetdir=/home/user/my_recon_surf_output  # equivalent to FreeSurfer's SUBJECT_DIR

# Create log directory (optional)
mkdir -p $targetdir/logs

# Run recon-surf
while read p ; do
  echo $p
  nohup ./recon-surf.sh --sid ${p} \
                        --sd $targetdir \
                        --py python3.8 > $targetdir/logs/out-${p}.log &
  sleep 3 
done < /home/user/my_mri_data/subject_list.txt

```

A dedicated subfolder will be used for each subject within the target directory. 
As the --t1 and --aparc_aseg_segfile flags are not set, a subfolder within the target directory named after each subject ($p) needs to exist and contain t1-weighted conformed image, 
mask and segmentations (as output by our FastSurfer segmentation networks, i.e. under /home/user/my_fastsurfeer_analysis/$p/mri/aparc.DKTatlas+aseg.deep.mgz, mask.mgz, and orig.mgz)). 
The directory will then be populated with the FreeSurfer file structure, including surfaces, statistics 
and labels file (equivalent to a FreeSurfer recon-all run). 
A log-file will additionally be  stored in the ```$targetdir/logs``` directory. If you do not need this, remove the corresponding redirect ( ```> $targetdir/logs/out-${p}.log```).


# Manual Edits

### Brainmask Edits

Currently, FastSurfer has only very limited functionality for manual edits due to missing entrypoints into the recon-surf script. Starting with FastSurfer v2.0.0 one frequently requested edit type (brainmask editing) is now possible, as the initial mask is created in the first segmentation stage. By running segmentation and surface processing in two steps, the mask can be edited in-between.

For a **Docker setup** one can:

1. Run segmentation only:
```
docker run --gpus=all --rm --name $CONTAINER_NAME \
                      -v $PATH_TO_IMAGE_DIR:$IMAGE_DIR \
                      -v $PATH_TO_OUTPUT_DIR:$OUTPUT_DIR \
                      --user $UID:$GID deepmi/fastsurfer:gpu-v2.0.0 \
                      --t1 $IMAGE_DIR/input.mgz \
                      --sd $OUTPUT_DIR \
                      --sid $SUBJECT_ID \
                      --seg_only
```
2. Modify the ```$PATH_TO_OUTPUT_DIR/$SUBJECT_ID/mri/mask.mgz``` file as required.
3. Run the following Docker command to run the surface processing pipeline:
```
docker run --rm --name $CONTAINER_NAME \
           -v $PATH_TO_OUTPUT_DIR:$OUTPUT_DIR \
           -v $PATH_TO_FS_LICENSE_DIR:$FS_LICENSE_DIR \
           --user $UID:$GID deepmi/fastsurfer:gpu-v2.0.0 \
           --sid $SUBJECT_ID  \
           --sd $OUTPUT_DIR/$SUBJECT_ID \
           --surf_only \
           --fs_license $FS_LICENSE_DIR/license_file
```

For a **local install** you can similarly:

1. Go to the FastSurfer directory, source FreeSurfer 7.3.2 and run the segmentation step:
```
cd $FASTSURFER_HOME
source $FREESURFER_HOME/SetUpFreeSurfer.sh
./run_fastsurfer.sh --t1 $IMAGE_DIR/input.mgz --sd $OUTPUT_DIR --sid $SUBJECT_ID --seg_only
```
2. Modify the ```$OUTPUT_DIR/$SUBJECT_ID/mri/mask.mgz``` file.
3. Run the surface pipeline:
```
./run_fastsurfer.sh --sd $OUTPUT_DIR --sid $SUBJECT_ID --fs_license $FS_LICENSE_DIR/license_file --surf_only
```
