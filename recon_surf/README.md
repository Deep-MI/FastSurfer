# Overview - recon-surf

This directory contains all information needed to run the surface-generation and processing part of FastSurfer. Within 
approximately 1-1.5 h (depending on processing parallelization and image quality) this pipeline provides a fast FreeSurfer 
alternative for cortical surface reconstruction, mapping of cortical labels and traditional point-wise and ROI thickness analysis.
The basis for the reconstruction pipeline is the accurate anatomical whole brain segmentation following the DKTatlas 
such as the one provided by the FastSurferCNN deep learning architecture.

# Usage
The *recon_surf* directory contains all the source code and binaries (mris_make_surfaces from the FreeSurfer dev version)
to run the analysis. In addition, a working distribution of __FreeSurfer__ (v6.0 or dev) is needed.

The main script is called __recon-surf.sh__ which accepts certain arguments via the command line.
List them by running the following command:

```bash
./recon-surf.sh --help
```

### Required arguments
* --sd: Output directory \$SUBJECTS_DIR (equivalent to FreeSurfer setup --> $SUBJECTS_DIR/sid/mri; $SUBJECTS_DIR/sid/surf ... will be created).
* --sid: Subject ID for directory inside \$SUBJECTS_DIR to be created ($SUBJECTS_DIR/sid/...)
* --t1: T1 full head input (not bias corrected). This must be conformed (dimensions: 256x256x256, voxel size: 1x1x1, LIA orientation, and data type UCHAR). Images can be conformed using FastSurferCNN's [conform.py](https://github.com/Deep-MI/FastSurfer/blob/master/FastSurferCNN/data_loader/conform.py) script (usage example: python3 FastSurferCNN/data_loader/conform.py -i <T1_input> -o <conformed_T1_output>).

### Optional arguments
* --seg: Global path with filename of segmentation (where and under which name to find it, must already exist). This must be conformed (dimensions: 256x256x256, voxel size: 1x1x1, and LIA orientation). FastSurferCNN's segmentations are conformed by default. Please ensure that segmentations produced otherwise are also conformed. Default location: $SUBJECTS_DIR/$sid/mri/aparc.DKTatlas+aseg.deep.mgz
* --fstess: Use mri_tesselate instead of marching cube (default) for surface creation
* --fsqsphere: Use FreeSurfer default instead of novel spectral spherical projection for qsphere
* --fsaparc: Use FS aparc segmentations in addition to DL prediction (slower in this case and usually the mapped ones from the DL prediction are fine)
* --surfreg: Run Surface registration with FreeSurfer (for cross-subject correspondance)
* --parallel: Run both hemispheres in parallel
* --threads: Set openMP and ITK threads to <int>

### Other
* --py: which python version to use. Default: python3.6

### Example 1: recon-surf on a single subject (subject1)

Given you want to analyze data for subject1 which is stored on your computer under /home/user/my_mri_data/subject1/orig.mgz, 
run the following command from the console (do not forget to source FreeSurfer!):

```bash
# Source FreeSurfer
export FREESURFER_HOME=/path/to/freesurfer/fs60
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Define data directory
datadir=/home/user/my_mri_data
segdir=/home/user/my_segmentation_data
targetdir=/home/user/my_recon_surf_output  # equivalent to FreeSurfer's SUBJECT_DIR

# Run recon-surf
./recon-surf.sh --sid subject1 \
                --sd $targetdir \
                --t1 $datadir/subject1/orig.mgz 
                --py python3.6

```

A subfolder within the target directory named after the subject (here: subject1) will automatically be created and populated
with the generated image, surface, statistics and labels file (equivalent to a FreeSurfer recon-all run). 

### Example 2: recon-surf on multiple subjects (using nohup)

In order to run recon-surf on a number of cases which are stored in the same directory, prepare a subjects_list.txt file listing the names line per line:
subject1\n
subject2\n
subject3\n
...
subject10\n

And invoke the following command (make sure you have enough ressources to run the given number of subjects in parallel!):

```bash
# Source FreeSurfer
export FREESURFER_HOME=/path/to/freesurfer/fs60
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
                        --t1 $datadir/${p}/orig.mgz \
                        --py python3.6 > $targetdir/logs/out-${p}.log &
  sleep 3 
done < /home/user/my_mri_data/subject_list.txt

```

A dedicated subfolder will be generated for each subject within the target directory. A log-file will additionally be 
stored in the $targetdir/logs directory. If you do not need this, remove the corresponding redirect (> $targetdir/logs/out-${p}.log).
