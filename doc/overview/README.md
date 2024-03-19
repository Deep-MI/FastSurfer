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
Notwithstanding module-specific limitations, resolution should be between 1mm and 0.7mm isotropic (slice thickness should not exceed 1.5mm). Preferred sequence is Siemens MPRAGE or multi-echo MPRAGE. GE SPGR should also work. See `--vox_size` flag for high-res behaviour.

![](../../images/teaser.png)


# Getting started
## Installation 
There are two ways to run FastSurfer (links are to installation instructions):

1. In a container ([Singularity](INSTALL.md#singularity) or [Docker](INSTALL.md#docker)) (OS: [Linux](INSTALL.md#linux), [Windows](INSTALL.md#windows), [MacOS on Intel](INSTALL.md#docker--currently-only-supported-for-intel-cpus-)),
2. As a [native install](INSTALL.md#native--ubuntu-2004-) (all OS for segmentation part). 

We recommended you use Singularity or Docker, especially if either is already installed on your system, because the images we provide on [DockerHub](https://hub.docker.com/r/deepmi/fastsurfer) conveniently include everything needed for FastSurfer, expect a [FreeSurfer license file](https://surfer.nmr.mgh.harvard.edu/fswiki/License). We have detailed, per-OS Installation instructions in the [INSTALL.md file](INSTALL.md).

## Usage

All installation methods use the `run_fastsurfer.sh` call interface (replace `*fastsurfer-flags*` with [FastSurfer Flag](FLAGS.md#required-arguments), which is the general starting point for FastSurfer. However, there are different ways to call this script depending on the installation, which we explain here:

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

   The `--no-home` flag tells singularity to not mount the home directory (see [Singularity README](../../Singularity/README.md#mounting-home) for more info).

   The `-B` flag is used to tell singularity, which folders FastSurfer can read and write to.
 
   See also Example 2 FastSurfer Singularity for a full singularity FastSurfer run command and [the Singularity README](../../Singularity/README.md#fastsurfer-singularity-image-usage) for details on more singularity flags.  

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
 
   See also example 1 fastSurfer-docker for a full FastSurfer run inside a Docker container and [the Docker README](../../Docker/README.md#docker-flags-) for more details on the docker flags including `--rm` and `--user`.

2. For a __native install__, you need to activate your FastSurfer environment (e.g. `conda activate fastsurfer_gpu`) and make sure you have added the FastSurfer path to your `PYTHONPATH` variable, e.g. `export PYTHONPATH=$(pwd)`. 

   You will then be able to run fastsurfer with `./run_fastsurfer.sh *fastsurfer-flags*`.

   See also example 3 i.e. native fastsurfer on subjectx with-parallel processing of hemis for an illustration of the commands to run the entire FastSurfer pipeline (FastSurferCNN + recon-surf) natively.

### FastSurfer_Flags
Please refer to [FASTSURFER_FLAGS](FLAGS.md).


### Examples
All the examples can be found here: [FASTSURFER_EXAMPLES](EXAMPLES.md)
- [Example 1: FastSurfer Docker](EXAMPLES.md#example-1-fastsurfer-docker)
- [Example 2: FastSurfer Singularity](EXAMPLES.md#example-2-fastsurfer-singularity)
- [Example 3: Native FastSurfer on subjectX with parallel processing of hemis](EXAMPLES.md#example-3-native-fastsurfer-on-subjectx-with-parallel-processing-of-hemis)
- [Example 4: FastSurfer on multiple subjects](EXAMPLES.md#example-4-fastsurfer-on-multiple-subjects)
- [Example 5: Quick Segmentation](EXAMPLES.md#example-5-quick-segmentation)
- [Example 6: Running FastSurfer on a SLURM cluster via Singularity](EXAMPLES.md#example-6-running-fastsurfer-on-a-slurm-cluster-via-singularity)

## Output files

Modules output can be found here: [FastSurfer_Output_Files](OUTPUT_FILES.md)
- [Segmentation module](OUTPUT_FILES.md#segmentation-module)
- [Cerebnet module](OUTPUT_FILES.md#cerebnet-module)
- [Surface module](OUTPUT_FILES.md#surface-module)

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
This is documented in READMEs in subfolders, for example: [whole brain segmentation only with FastSurferVINN](../../FastSurferCNN/README.md), [cerebellum sub-segmentation (in progress)](../../CerebNet/README.md) and [surface pipeline only (recon-surf)](../../recon_surf/README.md).

Specifically, the segmentation modules feature options for optimized parallelization of batch processing.

## FreeSurfer Downstream Modules

FreeSurfer provides several Add-on modules for downstream processing, such as subfield segmentation ( [hippocampus/amygdala](https://surfer.nmr.mgh.harvard.edu/fswiki/HippocampalSubfieldsAndNucleiOfAmygdala), [brainstrem](https://surfer.nmr.mgh.harvard.edu/fswiki/BrainstemSubstructures), [thalamus](https://freesurfer.net/fswiki/ThalamicNuclei) and [hypothalamus](https://surfer.nmr.mgh.harvard.edu/fswiki/HypothalamicSubunits) ) as well as [TRACULA](https://surfer.nmr.mgh.harvard.edu/fswiki/Tracula). We now provide symlinks to the required files, as FastSurfer creates them with a different name (e.g. using "mapped" or "DKT" to make clear that these file are from our segmentation using the DKT Atlas protocol, and mapped to the surface). Most subfield segmentations require `wmparc.mgz` and work very well with FastSurfer,  so feel free to run those pipelines after FastSurfer. TRACULA requires `aparc+aseg.mgz` which we now link, but have not tested if it works, given that [DKT-atlas](https://mindboggle.readthedocs.io/en/latest/labels.html) merged a few labels. You should source FreeSurfer 7.3.2 to run these modules. 

## Intended Use

This software can be used to compute statistics from an MR image for research purposes. Estimates can be used to aggregate population data, compare groups etc. The data should not be used for clinical decision support in individual cases and, therefore, does not benefit the individual patient. Be aware that for a single image, produced results may be unreliable (e.g. due to head motion, imaging artefacts, processing errors etc). We always recommend to perform visual quality checks on your data, as also your MR-sequence may differ from the ones that we tested. No contributor shall be liable to any damages, see also our software [LICENSE](../../LICENSE). 

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

