[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deep-MI/FastSurfer/blob/stable/Tutorial/Tutorial_FastSurferCNN_QuickSeg.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deep-MI/FastSurfer/blob/stable/Tutorial/Complete_FastSurfer_Tutorial.ipynb)

# Overview

This directory contains all information needed to run FastSurfer - a fast and accurate deep-learning based neuroimaging pipeline. This approach provides a full [FreeSurfer](https://freesurfer.net/) alternative for volumetric analysis (within 1 minute) and surface-based thickness analysis (within only around 1h run time). The whole pipeline consists of two main parts:

(i) FastSurferCNN/VINN - an advanced deep learning architecture capable of whole brain segmentation into 95 classes in under
1 minute, mimicking FreeSurfer’s anatomical segmentation and cortical parcellation (DKTatlas)

(ii) recon-surf - full FreeSurfer alternative for cortical surface reconstruction, mapping of cortical labels and traditional point-wise and ROI thickness analysis in approximately 60 minutes. For surface group analysis, sphere.reg is also generated automatically by default. To safe time, this can be turned off with the --no_surfreg flag.

Image input requirements are identical to FreeSurfer: good quality T1-weighted MRI acquired at 3T with a resolution between 1mm and 0.7mm isotropic (slice thickness should not exceed 1.5mm). Preferred sequence is Siemens MPRAGE or multi-echo MPRAGE. GE SPGR should also work. Sub-mm scans (e.g. .75 or .8mm isotropic) will be conformed to the highest resolution (smallest per-direction voxel size). This behaviour can be changed with the --vox_size flag.

Within this repository, we provide the code and Docker files for running FastSurferCNN/VINN (segmentation only) and recon-surf (surface pipeline only) independently from each other or as a whole pipeline (run_fastsurfer.sh, segmentation + surface pipeline). For each of these purposes, see the README.md's in the corresponding folders.

![](/images/teaser.png)

## Usage
There are three ways to run FastSurfer - (a) using Docker, (b) using Singularity, (c) as a native install. The recommended way is to use Docker or Singularity. If either is already installed on your system, there are only two commands to get you started (1. the download of a container image, and 2. running it). Installation instructions, especially for the more involved native install, can be found in [INSTALL](INSTALL.md).

(a) For __docker__, simply pull our official images from [Dockerhub](https://hub.docker.com/r/deepmi/fastsurfer) (```docker pull deepmi/fastsurfer:latest```). No other local installations are needed (FreeSurfer and everything else will be included, you only need to provide a [FreeSurfer license file](https://surfer.nmr.mgh.harvard.edu/fswiki/License)). __[Example 1](#example-1:-fastSurfer-docker)__ shows how to run FastSurfer inside a Docker container. Alternatively,
use the provided Dockerfiles in our Docker directory to build your own image (see the [README](docker/README.md) in the Docker directory). __Mac users__ need to increase docker memory to 15 GB by overwriting the settings under Docker Desktop --> Preferences --> Resources --> Advanced (slide the bar under Memory to 15 GB; see: [docker for mac](https://docs.docker.com/docker-for-mac/) for details).

(b) For __singularity__, create a Singularity image from our official Dockerhub images (```singularity build fastsurfer-latest.sif docker://deepmi/fastsurfer:latest```). See __[Example 2](#example-2:-fastSurfer-singularity)__ for a singularity FastSurfer run command. Additionally, the [README](singularity/README.md) in the Singularity directory contains detailed directions for building your own Singularity images from Docker. 

(c) For a __native install__ on a modern linux (we tested Ubuntu 20.04), download this github repository (use git clone or download as zip and unpack) for the necessary source code and python scripts. You also need to have the necessary Python 3 libraries installed (see __requirements.txt__) as well as bash-4.0 or higher (when using pip, upgrade pip first as older versions will fail). This is already enough to generate the whole-brain segmentation using FastSurferCNN (see the README.md in the FastSurferCNN directory for the exact commands). In order to run the whole FastSurfer pipeline (for surface creation etc.) locally on your machine, a working version of __FreeSurfer__ (v7.3.2, [https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads](https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads)) is required to be pre-installed and sourced. See [INSTALL](INSTALL.md) for detailled installation instructions and [Example 3](#example-3:-native-fastSurfer-on-subjectX-(with-parallel-processing-of-hemis)) or [Example 4](#example-4:-native-fastSurfer-on-multiple-subjects) for an illustration of the commands to run the entire FastSurfer pipeline (FastSurferCNN + recon-surf) natively.

In the native install, the main script called __run_fastsurfer.sh__ can be used to run both FastSurferCNN and recon-surf sequentially on a given subject. There are several options which can be selected and set via the command line. 
List them by running the following command:
```bash
./run_fastsurfer.sh --help
```

### Required arguments
* `--sd`: Output directory \$SUBJECTS_DIR (equivalent to FreeSurfer setup --> $SUBJECTS_DIR/sid/mri; $SUBJECTS_DIR/sid/surf ... will be created).
* `--sid`: Subject ID for directory inside \$SUBJECTS_DIR to be created ($SUBJECTS_DIR/sid/...)
* `--t1`: T1 full head input (not bias corrected, global path). The network was trained with conformed images (UCHAR, 256x256x256, 1-0.7 mm voxels and standard slice orientation). These specifications are checked in the run_prediction.py script and the image is automatically conformed if it does not comply.

### Required for docker when running surface module
* `--fs_license`: Path to FreeSurfer license key file. Register (for free) at https://surfer.nmr.mgh.harvard.edu/registration.html to obtain it if you do not have FreeSurfer installed so far. Strictly necessary if you use Docker, optional for local install (your local FreeSurfer license will automatically be used). The license file is usually located in $FREESURFER_HOME/license.txt or $FREESURFER_HOME/.license .

### Segmentation pipeline arguments (optional)
* `--seg_only`: only run FastSurferCNN (generate segmentation, do not run the surface pipeline)
* `--main_segfile`: Global path with filename of segmentation (where and under which name to store it). The file can be in mgz, nii, or nii.gz format. Default location: $SUBJECTS_DIR/$sid/mri/aparc.DKTatlas+aseg.deep.mgz
* `--seg_log`: Name and location for the log-file for the segmentation (FastSurferCNN). Default: $SUBJECTS_DIR/$sid/scripts/deep-seg.log
* `--viewagg_device`: Define where the view aggregation should be run on. Can be "auto" or a device (see --device). By default, the program checks if you have enough memory to run the view aggregation on the gpu. The total memory is considered for this decision. If this fails, or you actively overwrote the check with setting with "cpu" view agg is run on the cpu. Equivalently, if you pass a different device, view agg will be run on that device (no memory check will be done).
* `--device`: Select device for NN segmentation (_auto_, _cpu_, _cuda_, _cuda:<device_num>_), where cuda means Nvidia GPU, you can select which one e.g. "cuda:1". Default: "auto", check GPU and then CPU
* `--batch`: Batch size for inference. Default: 1. 
* `--vol_segstats`: Additionally return volume-based aparc.DKTatlas+aseg statistics for DL-based segmentation (does not  require surfaces). Can be used in combination with `--seg_only` in which case recon-surf only runs till CC is added.
* `--aparc_aseg_segfile <filename>`: Name of the segmentation file, which includes the aparc+DKTatlas-aseg segmentations. If not provided, this intermediate DL-based segmentation will not be stored, but only the merged segmentation will be stored (see --main_segfile <filename>). Requires an ABSOLUTE Path! Default location: \$SUBJECTS_DIR/\$sid/mri/aparc.DKTatlas+aseg.deep.mgz

### Surface pipeline arguments (optional)
* `--surf_only`: only run the surface pipeline recon_surf. The segmentation created by FastSurferCNN must already exist in this case.
* `--fstess`: Use mri_tesselate instead of marching cube (default) for surface creation
* `--fsqsphere`: Use FreeSurfer default instead of novel spectral spherical projection for qsphere
* `--fsaparc`: Use FS aparc segmentations in addition to DL prediction (slower in this case and usually the mapped ones from the DL prediction are fine)
* `--parallel`: Run both hemispheres in parallel
* `--threads`: Set openMP and ITK threads to <int>
* `--no_surfreg`: Skip surface registration with FreeSurfer (if only stats are needed), for the sake of speed. 
* `--no_fs_T1`: Do not generate T1.mgz (normalized nu.mgz included in standard FreeSurfer output) and create brainmask.mgz directly from norm.mgz instead. Saves 1:30 min.

### Other
* `--vox_size <0.7-1|min>`  Forces processing at a specific voxel size.
                            If a number between 0.7 and 1 is specified (below
                            is experimental) the T1w image is conformed to
                            that voxel size and processed.
                            If "min" is specified (default), the voxel size is
                            read from the size of the minimal voxel size
                            (smallest per-direction voxel size) in the T1w
                            image:
                              If the minimal voxel size is bigger than 0.98mm,
                                the image is conformed to 1mm isometric.
                              If the minimal voxel size is smaller or equal to
                                0.98mm, the T1w image will be conformed to
                                isometric voxels of that voxel size.
                            The voxel size (whether set manually or derived)
                            determines whether the surfaces are processed with
                            highres options (below 1mm) or not.
* `--py`: Command for python, used in both pipelines. Default: python3.8
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
                      --parallel
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

Note, that the paths following `--fs_license`, `--t1`, and `--sd` are __inside__ the container, not global paths on your system, so they should point to the places where you mapped these paths above with the `-v` arguments (part after colon). 

A directory with the name as specified in `--sid` (here subjectX) will be created in the output directory if it does not exist. So in this example output will be written to /home/user/my_fastsurfer_analysis/subjectX/ . Make sure the output directory is empty, to avoid overwriting existing files. 

If you do not have a GPU, you can also run our CPU-Docker with very similar commands. See [Docker/README.md](Docker/README.md) for more details.


### Example 2: FastSurfer Singularity
After building the Singularity image (see instructions in ./Singularity/README.md), you also need to register at the FreeSurfer website (https://surfer.nmr.mgh.harvard.edu/registration.html) to acquire a valid license (for free) - same as when using Docker. This license needs to be passed to the script via the `--fs_license` flag. This is not necessary if you want to run the segmentation only.

To run FastSurfer on a given subject using the Singularity image with GPU access, execute the following commands from a directory where you want to store singularity images. This will create a singularity image from our Dockerhub image and execute it:

```bash
# 1. Build the singularity image (if it does not exist)
singularity build fastsurfer-gpu.sif docker://deepmi/fastsurfer

# 2. Run command
singularity exec --nv -B /home/user/my_mri_data:/data \
                      -B /home/user/my_fastsurfer_analysis:/output \
                      -B /home/user/my_fs_license_dir:/fs_license \
                       ./fastsurfer-gpu.sif \
                       /fastsurfer/run_fastsurfer.sh \
                      --fs_license /fs_license/license.txt \
                      --t1 /data/subjectX/t1-weighted.nii.gz \
                      --sid subjectX --sd /output \
                      --parallel
```

Singularity Flags:
* The `--nv` flag is used to access GPU resources. 
* The `-B` commands mount your data, output, and directory with the FreeSurfer license file into the Singularity container. Inside the container these are visible under the name following the colon (in this case /data, /output, and /fs_license). 

FastSurfer Flags:
* The `--fs_license` points to your FreeSurfer license which needs to be available on your computer in the my_fs_license_dir that was mapped above. 
* The `--t1` points to the t1-weighted MRI image to analyse (full path, with mounted name inside docker: /home/user/my_mri_data => /data)
* The `--sid` is the subject ID name (output folder name)
* The `--sd` points to the output directory (its mounted name inside docker: /home/user/my_fastsurfer_analysis => /output)
* The `--parallel` activates processing left and right hemisphere in parallel

Note, that the paths following `--fs_license`, `--t1`, and `--sd` are __inside__ the container, not global paths on your system, so they should point to the places where you mapped these paths above with the `-v` arguments (part after colon).

A directory with the name as specified in `--sid` (here subjectX) will be created in the output directory. So in this example output will be written to /home/user/my_fastsurfer_analysis/subjectX/ . Make sure the output directory is empty, to avoid overwriting existing files. 

You can run the Singularity equivalent of CPU-Docker by building a Singularity image from the CPU-Docker image and excluding the `--nv` argument in your Singularity exec command.


### Example 3: Native FastSurfer on subjectX (with parallel processing of hemis)

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
                    --parallel --threads 4
```

The output will be stored in the $fastsurferdir (including the aparc.DKTatlas+aseg.deep.mgz segmentation under $fastsurferdir/subjectX/mri (default location)). Processing of the hemispheres will be run in parallel (--parallel flag). Omit this flag to run the processing sequentially.


### Example 4: Native FastSurfer on multiple subjects

In order to run FastSurfer on multiple cases which are stored in the same directory, prepare a subjects_list.txt file listing the names line by line:
subject1\n
subject2\n
subject3\n
...
subject10\n

And invoke the following command (make sure you have enough ressources to run the given number of subjects in parallel!):

```bash
export FREESURFER_HOME=/path/to/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

cd /home/user/FastSurfer
datadir=/home/user/my_mri_data
fastsurferdir=/home/user/my_fastsurfer_analysis
mkdir -p $fastsurferdir/logs # create log dir for storing nohup output log (optional)

while read p ; do
  echo $p
  nohup ./run_fastsurfer.sh --t1 $datadir/$p/t1-weighted.nii.gz
                            --sid $p --sd $fastsurferdir > $fastsurferdir/logs/out-${p}.log &
  sleep 90s 
done < ./data/subjects_list.txt
```

### Example 5 Quick Segmentation

For many applications you won't need the surfaces. You can run only the segmentation (in 1 minute on a GPU) via

```bash
./run_fastsurfer.sh --t1 $datadir/subject1/t1-weighted.nii.gz \
                    --main_segfile $ouputdir/subject1/aparc.DKTatlas+aseg.deep.mgz \
                    --conformed_name $ouputdir/subject1/conformed.mgz \
                    --parallel --threads 4 --seg_only
```

This will produce the segmentation in a conformed space (just as FreeSurfer would do). It also writes the conformed image that fits the segmentation.
Conformed means that the image will be isotropic in LIA orientation. 
It will furthermore output a brain mask (mask.mgz) and a simplified segmentation file (aseg.auto_noCCseg.mgz). 


Alternatively - but this requires a FreeSurfer install - you can get mask and also statistics after insertion of the corpus callosum by adding ```--vol_segstats``` in the run_fastsurfer.sh command:

```bash
./run_fastsurfer.sh --t1 $datadir/subject1/t1-weighted.nii.gz \
                    --main_segfile $ouputdir/subject1/aparc.DKTatlas+aseg.deep.mgz \
                    --conformed_name $ouputdir/subject1/conformed.mgz \
                    --parallel --threads 4 --seg_only --vol_segstats
```

The above ```run_fastsurfer.sh``` commands can also be called from the Docker or Singularity images by passing the flags and adjusting input and output directories to the locations inside the containers (where you mapped them via the -v flag in Docker or -B in Singularity). 

```bash
# Docker - segmentation only
docker run --gpus all -v $datadir:/data \
                      -v $outputdir:/output \
                      --rm --user $(id -u):$(id -g) deepmi/fastsurfer:latest \
                      --t1 /data/subject1/t1-weighted.nii.gz \
                      --main_segfile /ouput/subject1/aparc.DKTatlas+aseg.deep.mgz \
                      --conformed_name $ouputdir/subject1/conformed.mgz \
                      --parallel --threads 4 --seg_only
                      
# Docker - segmentation and statistics (fs-license required)
docker run --gpus all -v $datadir:/data \
                      -v $outputdir:/output \
                      -v /home/user/my_fs_license_dir:/fs_license \
                      --rm --user $(id -u):$(id -g) deepmi/fastsurfer:latest \
                      --fs_license /fs_license/license.txt \
                      --t1 /data/subject1/t1-weighted.nii.gz \
                      --main_segfile $ouputdir/subject1/aparc.DKTatlas+aseg.deep.mgz \
                      --conformed_name $ouputdir/subject1/conformed.mgz \
                      --parallel --threads 4 --seg_only --vol_segstats
```

## System Requirements

Recommendation: At least 8GB CPU RAM and 8GB NVIDIA GPU RAM ``--viewagg_device gpu``  

Minimum: 7 GB CPU RAM and 2 GB GPU RAM ``--viewagg_device cpu --vox_size 1``

Minimum CPU-only: 8 GB CPU RAM (much slower, not recommended) ``--device cpu --vox_size 1`` 

### Minimum Requirements:

|       | --viewagg_device | Min GPU (in GB) | Min CPU (in GB) |
|:------|------------------|----------------:|----------------:|
| 1mm   | gpu              |               5 |               5 |
| 1mm   | cpu              |               2 |               7 |
| 0.8mm | gpu              |               8 |               6 |
| 0.8mm | cpu              |               3 |               9 |
| 0.7mm | gpu              |               8 |               6 |
| 0.7mm | cpu              |               3 |               9 |


## FreeSurfer Downstream Modules

FreeSurfer provides several Add-on modules for downstream processing, such as subfield segmentation ( [hippocampus/amygdala](https://surfer.nmr.mgh.harvard.edu/fswiki/HippocampalSubfieldsAndNucleiOfAmygdala), [brainstrem](https://surfer.nmr.mgh.harvard.edu/fswiki/BrainstemSubstructures), [thalamus](https://freesurfer.net/fswiki/ThalamicNuclei) and [hypothalamus](https://surfer.nmr.mgh.harvard.edu/fswiki/HypothalamicSubunits) ) as well as [TRACULA](https://surfer.nmr.mgh.harvard.edu/fswiki/Tracula). We now provide symlinks to the required files, as FastSurfer creates them with a different name (e.g. using "mapped" or "DKT" to make clear that these file are from our segmentation using the DKT Atlas protocol, and mapped to the surface). Most subfield segmentations require `wmparc.mgz` and work very well with FastSurfer,  so feel free to run those pipelines after FastSurfer. TRACULA requires `aparc+aseg.mgz` which we now link, but have not tested if it works, given that [DKT-atlas](https://mindboggle.readthedocs.io/en/latest/labels.html) merged a few labels. You should source FreeSurfer 7.3.2 to run these modules. 

## Intended Use

This software can be used to compute statistics from an MR image for research purposes. Estimates can be used to aggregate population data, compare groups etc. The data should not be used for clinical decision support in individual cases and, therefore, does not benefit the individual patient. Be aware that for a single image, produced results may be unreliable (e.g. due to head motion, imaging artefacts, processing errors etc). We always recommend to perform visual quality checks on your data, as also your MR-sequence may differ from the ones that we tested. No contributor shall be liable to any damages, see also our software [LICENSE](LICENSE). 

## References

If you use this for research publications, please cite:

Henschel L, Conjeti S, Estrada S, Diers K, Fischl B, Reuter M, FastSurfer - A fast and accurate deep learning based neuroimaging pipeline, NeuroImage 219 (2020), 117012. https://doi.org/10.1016/j.neuroimage.2020.117012

Henschel L*, Kügler D*, Reuter M. (*co-first). FastSurferVINN: Building Resolution-Independence into Deep Learning Segmentation Methods - A Solution for HighRes Brain MRI. NeuroImage 251 (2022), 118933. http://dx.doi.org/10.1016/j.neuroimage.2022.118933

Stay tuned for updates and follow us on Twitter: https://twitter.com/deepmilab

## Acknowledgements
The recon-surf pipeline is largely based on FreeSurfer 
https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferMethodsCitation
