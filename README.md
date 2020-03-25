# Overview

This directory contains all information needed to run FastSurfer - a fast and accurate deep-learning based neuroimaging pipeline.  This approach provides a full FreeSurfer alternative for volumetric analysis (within
1  minute)  and  surface-based  thickness  analysis  (within  only  around  1h  run  time). It consists of two main parts:
(i) FastSurferCNN - an advanced deep learning architecture capable of whole brain segmentation into 95 classes in under
1 minute, mimicking FreeSurfer’s anatomical segmentation and cortical parcellation (DKTatlas)
(ii) recon-surf - full  FreeSurfer  alternative for cortical surface reconstruction, mapping of cortical labels and traditional point-wise and ROI thickness analysis in approximately 60 minutes.


![](/images/teaser.png)


## Usage
The *FastSurferCNN* and *recon_surf* directories contain all the source code and modules needed to run the scripts. A list of python libraries used within the code can be found in __requirements.txt__. In addition, a working version of __FreeSurfer__ (v6.0 or dev) is needed to run recon-surf.
The main script is called __run_fastsurfer.sh__ which can be used to run both FastSurferCNN and recon-surf sequentially on a given subject. There are a number of options which can be selected and set via the command line.
List them by running the following command:
```bash
./run_fastsurfer.sh --help
```

### Required arguments
* --fs_license: Path to FreeSurfer license key file. Register (for free) at https://surfer.nmr.mgh.harvard.edu/registration.html to obtain it if you do not have FreeSurfer installed so far.
* --sd: Output directory \$SUBJECTS_DIR (equivalent to FreeSurfer setup --> $SUBJECTS_DIR/sid/mri; $SUBJECTS_DIR/sid/surf ... will be created).
* --sid: Subject ID for directory inside \$SUBJECTS_DIR to be created ($SUBJECTS_DIR/sid/...)
* --t1: T1 full head input (not bias corrected). The network was trained with conformed images (UCHAR, 256x256x256, 1 mm voxels and standard slice orientation). These specifications are checked in the eval.py script and the image is automatically conformed if it does not comply.
* --seg: Name and location of segmentation (where and under which name to store it)

### Network specific arguments (optional)
* --weights_sag: Pretrained weights of sagittal network. Default: ../checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl
* --weights_ax: Pretrained weights of axial network. Default: ../checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl
* --weights_cor: Pretrained weights of coronal network. Default: ../checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl
* -- seg_log: Name and location for the log-file for the segmentation (FastSurferCNN)
* --clean_seg: Flag to clean up FastSurferCNN segmentation
* --no_cuda: Flag to disable CUDA usage in FastSurferCNN (no GPU usage, inference on CPU)
* --batch: Batch size for inference. Default: 16
* --order: Order of interpolation for mri_convert T1 before segmentation (0=nearest, 1=linear(default), 2=quadratic, 3=cubic)

### Surface pipeline arguments (optional)
* --mc: Switch on marching cube for surface creation
* --qspec: Switch on spectral spherical projection for qsphere
* --nofsaparc: Skip FS aparc segmentations and ribbon for speedup
* --surfreg: Run Surface registration with FreeSurfer (for cross-subject correspondence)
* --parallel: Run both hemispheres in parallel
* --threads: Set openMP and ITK threads to <int>

### Other
* --py: which python version to use. Default: python3.6
* --dev: Flag to set if FreeSurfer dev version is used
    

### Example 1: FastSurfer on subject1

Given you want to analyze data for subject1 which is stored on your computer under /home/user/my_mri_data/subject1/orig.mgz, run the following command from the console (do not forget to source FreeSurfer!):

```bash
# Source FreeSurfer
export FREESURFER_HOME=/path/to/freesurfer/fs60
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Define data directory
datadir=/home/user/my_mri_data
fastsurferdir=/home/user/my_fastsurfer_analysis

# Run FastSurfer
./run_fastsurfer.sh --fs_license /path/to/freesurfer/fs60/.license \
                    --t1 $datadir/subject1/orig.mgz \
                    --seg $fastsurferdir/subject1/aparc.DKTatlas+aseg.deep.mgz \
                    --sid subject1 --sd $fastsurferdir \
                    --mc --qspec --nofsaparc --parallel --threads 4
```

The output will be stored in the $fastsurferdir (including the aparc.DKTatlas+aseg.deep.mgz segmentation).

### Example 2: FastSurfer on multiple subjects (parallel processing)

In order to run FastSurfer on a number of cases which are stored in the same directory, prepare a subjects_list.txt file listing the names line per line:
subject1\n
subject2\n
subject3\n
...
subject10\n

And invoke the following command (make sure you have enough ressources to run the given number of subjects in parallel!):

```bash
export FREESURFER_HOME=/path/to/freesurfer/fs60
source $FREESURFER_HOME/SetUpFreeSurfer.sh

cd /home/user/FastSurfer
datadir=/home/user/my_mri_data
fastsurferdir=/home/user/my_fastsurfer_analysis
mkdir $fastsurferdir/logs # create log dir for storing nohup output log (optional)

while read p ; do
  echo $p
  nohup ./run_fastsurfer.sh --fs_license /path/to/freesurfer/fs60/.license \
                            --t1 $datadir/$p/orig.mgz \
                            --seg $fastsurferdir/$p/aparc.DKTatlas+aseg.deep.mgz \
                            --sid $p --sd $fastsurferdir \
                            --mc --qspec --nofsaparc > $fastsurferdir/logs/out-${p}.log &
  sleep 90s 
done < ./data/subjects_list.txt
```

### Example 3: FastSurfer inside Docker
After building the Docker (see instructions in ./Docker/README.md), you do not need to have a separate installation of FreeSurfer on your computer (included in the Docker). However, you need to register at the FreeSurfer website (https://surfer.nmr.mgh.harvard.edu/registration.html) to acquire a valid license (for free).

To run FastSurfer on a given subject using the provided Docker, execute the following command:

```bash
docker run --gpus all -v /home/user/my_mri_data:/data \
                      -v /home/user/my_fastsurfer_analysis:/output \
                      -v /home/user/my_fs_license_dir:/fs60 \
                      --rm --user XXXX fastsurfer:gpu \
                      --fs_license /fs60/.license \
                      --t1 /data/subject2/orig.mgz \
                      --seg /output/subject2/aparc.DKTatlas+aseg.deep.mgz \
                      --sid subject2 --sd /output \
                      --mc --qspec --nofsaparc --parallel
```

* The fs_license points to your FreeSurfer license which needs to be available on your computer (e.g. in the /home/user/my_fs_license_dir folder). 
* The --gpus flag is used to access GPU resources. With it you can also specify how many GPUs to use. In the example above, _all_ will use all available GPUS. To use a single one (e.g. GPU 0), set --gpus device=0. To use multiple specific ones (e.g. GPU 0, 1 and 3), set --gpus '"device=0,1,3"'.
* The -v command mounts your data (and output) directory into the docker image. Inside it is visible under the name following the colon (in this case /data or /output).
* The --rm flag takes care of removing the container once the analysis finished. 
* The --user XXXX part should be changed to the appropriate user id (a four digit number; can be checked with the command "id -u" on linux systems). All generated files will then belong to the specified user. Without the flag, the docker container will be run as root.


Within this repository, we further provide the code and Docker files for running FastSurferCNN and recon-surf independently from each other. For each of these purposes, see the README.md's in the corresponding folders.

## Acknowledgements
The recon-surf pipeline is largely based on FreeSurfer including the use of one binary (mris_make_surfaces) from the dev version. 
https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferMethodsCitation
