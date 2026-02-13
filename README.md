[![DOI](https://zenodo.org/badge/211859022.svg)](https://zenodo.org/badge/latestdoi/211859022)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deep-MI/FastSurfer/blob/stable/Tutorial/Tutorial_FastSurferCNN_QuickSeg.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deep-MI/FastSurfer/blob/stable/Tutorial/Complete_FastSurfer_Tutorial.ipynb)

<!-- start of content -->
Welcome to FastSurfer!
======================

Overview
--------
This README contains all information needed to run FastSurfer - a fast and accurate deep-learning based neuroimaging pipeline. FastSurfer provides a fully compatible [FreeSurfer](https://freesurfer.net/) alternative for volumetric analysis (within minutes) and surface-based thickness analysis (within only around 1h run time). 
FastSurfer is transitioning to sub-millimeter resolution support throughout the pipeline.

The FastSurfer pipeline consists of two main parts for segmentation and surface reconstruction.  

- the segmentation sub-pipeline (`seg`) employs advanced deep learning networks for fast, accurate segmentation and volumetric calculation of the whole brain and selected substructures.
- the surface sub-pipeline (`recon-surf`) reconstructs cortical surfaces, maps cortical labels and performs a traditional point-wise and ROI thickness analysis. 

### Segmentation Modules 
- approximately 5 minutes (GPU), `--seg_only` only runs this part. 
 
Modules (all run by default):
1. `asegdkt:` [FastSurferVINN](FastSurferCNN/README.md) for whole brain segmentation (deactivate with `--no_asegdkt`)
   - the core, outputs anatomical segmentation and cortical parcellation and statistics of 95 classes, mimics FreeSurfer’s DKTatlas.
   - requires a T1w image ([notes on input images](#requirements-to-input-images)), supports high-res (up to 0.7mm, experimental beyond that).
   - performs bias-field correction and calculates volume statistics corrected for partial volume effects (skipped if `--no_biasfield` is passed).
2. `cc`: [CorpusCallosum](CorpusCallosum/README.md) for corpus callosum segmentation and shape analysis (deactivate with `--no_cc`)
   - requires `asegdkt_segfile` (segmentation) and conformed mri (orig.mgz), outputs CC segmentation, thickness, and shape metrics.
   - standardizes brain orientation based on AC/PC landmarks (orient_volume.lta).
3. `cereb:` [CerebNet](CerebNet/README.md) for cerebellum sub-segmentation (deactivate with `--no_cereb`)
   - requires `asegdkt_segfile`, outputs cerebellar sub-segmentation with detailed WM/GM delineation.
   - requires a T1w image ([notes on input images](#requirements-to-input-images)), which will be resampled to 1mm isotropic images (no native high-res support).
   - calculates volume statistics corrected for partial volume effects (skipped if `--no_biasfield` is passed).
4. `hypothal`: [HypVINN](HypVINN/README.md) for hypothalamus subsegmentation (deactivate with `--no_hypothal`)
   - outputs a hypothalamic subsegmentation including 3rd ventricle, c. mammilare, fornix and optic tracts.
   - a T1w image is highly recommended ([notes on input images](#requirements-to-input-images)), supports high-res (up to 0.7mm, but experimental beyond that).
   - allows the additional passing of a T2w image with `--t2 <path>`, which will be registered to the T1w image (see `--reg_mode` option).
   - calculates volume statistics corrected for partial volume effects based on the T1w image (skipped if `--no_biasfield` is passed).

### Surface reconstruction
- approximately 60-90 minutes, `--surf_only` runs only [the surface part](recon_surf/README.md).
- supports high-resolution images (up to 0.7mm, experimental beyond that).
- requires a FreeSurfer license file as it uses some FreeSurfer binaries internally.
- requires outputs of the `asegdkt` and the `cc` modules as a prerequisite (can be included in the same run).

<!-- start of image requirements -->
### Requirements to input images
All pipeline parts and modules require good quality MRI images, preferably from a 3T MR scanner.
FastSurfer expects a similar image quality as FreeSurfer, so what works with FreeSurfer should also work with FastSurfer. 
Notwithstanding module-specific limitations, resolution should be between 1mm and 0.7mm isotropic (slice thickness should not exceed 1.5mm). Preferred sequence is Siemens MPRAGE or multi-echo MPRAGE. GE SPGR should also work. See `--vox_size` flag for high-res behaviour.
<!-- end of image requirements -->

![](doc/images/teaser.png)

<!-- start of getting started -->
Getting started
---------------

### Installation 
There are three ways to run FastSurfer (links are to installation instructions):

1. For Linux and Windows users, we recommend running FastSurfer in a container [Singularity/Apptainer](doc/overview/INSTALL.md#singularity) or [Docker](doc/overview/INSTALL.md#docker): (OS: [Linux](doc/overview/INSTALL.md#linux), [Windows](doc/overview/INSTALL.md#windows), [MacOS on Intel](doc/overview/INSTALL.md#docker-currently-only-supported-for-intel-cpus)),
2. for macOS users, we recommend [installing the FastSurfer package](doc/overview/INSTALL.md#package), and
3. for developers, the native install gives full control (only documented for [Linux](doc/overview/INSTALL.md#native-ubuntu-2004-or-ubuntu-2204)).

The images we provide on [DockerHub](https://hub.docker.com/r/deepmi/fastsurfer) conveniently include everything needed for FastSurfer. You will also need a [FreeSurfer license](https://surfer.nmr.mgh.harvard.edu/fswiki/License) file for the [Surface pipeline](#surface-reconstruction). We have detailed per-OS Installation instructions in the [INSTALL.md](doc/overview/INSTALL.md) file.

### Usage
All installation methods use the `run_fastsurfer.sh` call interface (replace the placeholder `<*fastsurfer-flags*>` with [FastSurfer flags](doc/scripts/RUN_FASTSURFER.md#required-arguments)), which is the general starting point for FastSurfer. However, there are different ways to call this script depending on the installation, which we explain here:

1. For container installations, you need to set up the container (`<*singularity-flags*>` or `<*docker-flags*>`) in addition to the `<*fastsurfer-flags*>`:  
   1. For __Singularity__, the syntax is

      ```bash
      singularity run <*singularity-flags*> \
                      fastsurfer.sif \
                      <*fastsurfer-flags*>
      ```
      This command has two placeholders for flags: `<*singularity-flags*>` and `<*fastsurfer-flags*>`.
      `<*singularity-flags*>` [set up the singularity environment](doc/overview/SINGULARITY.md), `<*fastsurfer-flags*>` include the options that determine the [behavior of FastSurfer](doc/scripts/RUN_FASTSURFER.md):
      ### Basic FastSurfer Flags
      
      - `--t1`: the path to the image to process.
      - `--sd`: the path to the "Subjects Directory", where all results will be stored.
      - `--sid`: the identified for the results for this image (folder inside "Subjects Directory").
      - `--fs_license`: path to the FreeSurfer license file.
      
      All options are explained in detail in the [run_fastsurfer.sh documentation](doc/scripts/RUN_FASTSURFER.md).  

      An example for a simple full FastSurfer-Singularity command is
      ```bash
      singularity run --nv \
                      -B $HOME/my/mri_data \
                      -B $HOME/my/fastsurfer_analysis \
                      -B /software/freesurfer/license.txt \
                      fastsurfer.sif \
                      --t1 $HOME/my/mri_data/participant1/image1.nii.gz \
                      --sd $HOME/my/fastsurfer_analysis \
                      --sid part1_img1 \
                      --fs_license /software/freesurfer/license.txt
      ```

      See also __[Example 1](doc/overview/EXAMPLES.md#example-1-fastsurfer-singularity-or-apptainer)__ for a full singularity FastSurfer run command and [the Singularity documentation](doc/overview/SINGULARITY.md#fastsurfer-singularity-image-usage) for details on more singularity flags and how to create the `fastsurfer.sif` file.

   2. For __docker__, the syntax is
      ```bash
      docker run <*docker-flags*> \
                 deepmi/fastsurfer:<device>-v<version> \
                 <*fastsurfer-flags*>
      ```
      
      The options for `<*docker-flags*>` and [`<*fastsurfer-flags*>`](README.md#basic-fastsurfer-flags) follow very similar patterns as for Singularity ([but the names of `<*docker-flags*>` are different](Docker/README.md#docker-flags)).
 
      __[Example 2](doc/overview/EXAMPLES.md#example-2-fastsurfer-docker)__ also details a full FastSurfer run inside a Docker container and [the Docker documentation](tools/Docker/README.md#docker-flags) for more details on `*docker flags*` and the naming of docker images (`<device>-v<version>`).

2. For a __macOS package install__, start FastSurfer from Applications and call the `run_fastsurfer.sh` FastSurfer script with [FastSurfer flags](doc/scripts/RUN_FASTSURFER.md#required-arguments) from the terminal that is opened for you.

3. For a __native install__, call the `run_fastsurfer.sh` FastSurfer script directly. Your FastSurfer python/conda environment needs to be [set up](doc/overview/INSTALL.md#native-ubuntu-2004-or-ubuntu-2204) and activated.

   ```bash
   # activate fastsurfer environment
   conda activate fastsurfer
   
   /path/to/fastsurfer/run_fastsurfer.sh <*fastsurfer-flags*>
   ```

   In addition to the [Basic Flags](README.md#basic-fastsurfer-flags), note that you may need to use `--py python3.12` to specify your python version, see [FastSurfer flags for more details](doc/scripts/RUN_FASTSURFER.md#required-arguments).


   [Example 3](doc/overview/EXAMPLES.md#example-3-native-fastsurfer-on-subjectx-with-parallel-processing-of-hemis) also illustrates the running the FastSurfer pipeline natively.

<!-- start of examples -->
Examples
--------
The documentation includes [6 detailed Examples](doc/overview/EXAMPLES.md) on how to use FastSurfer. 
- [Example 1: FastSurfer Singularity](doc/overview/EXAMPLES.md#example-1-fastsurfer-singularity-or-apptainer)
- [Example 2: FastSurfer Docker](doc/overview/EXAMPLES.md#example-2-fastsurfer-docker)
- [Example 3: Native FastSurfer on subjectX with parallel processing of hemis](doc/overview/EXAMPLES.md#example-3-native-fastsurfer-on-subjectx-with-parallel-processing-of-hemis)
- [Example 4: FastSurfer on multiple subjects](doc/overview/EXAMPLES.md#example-4-fastsurfer-on-multiple-subjects)
- [Example 5: Quick Segmentation](doc/overview/EXAMPLES.md#example-5-quick-segmentation)
- [Example 6: Running FastSurfer on a SLURM cluster via Singularity](doc/overview/EXAMPLES.md#example-6-running-fastsurfer-on-a-slurm-cluster-via-singularity)

Output files
------------
Modules output can be found here: [FastSurfer_Output_Files](doc/overview/OUTPUT_FILES.md)
- [Segmentation module](doc/overview/OUTPUT_FILES.md#segmentation-module)
- [Corpus Callosum module](doc/overview/OUTPUT_FILES.md#corpus-callosum-module)
- [Cerebnet module](doc/overview/OUTPUT_FILES.md#cerebnet-module)
- [HypVINN module](doc/overview/OUTPUT_FILES.md#hypvinn-module)
- [Surface module](doc/overview/OUTPUT_FILES.md#surface-module)

<!-- start of system requirements -->
System Requirements
-------------------

### Recommendation
- intel or AMD CPU (6 or more cores)
- 16 GB system memory
- nVidia graphics card (2016 or newer) 
- 12 GB graphics memory

FastSurfer supports multiple hardware acceleration modes: fully CPU (`--device cpu`), partial GPU 
(`--device cuda --viewagg_device cpu`) and fully GPU (`--device cuda`). By default, FastSurfer will try to pick the best
option. These modes require different system and video memory capacities, see the table below. 

| Voxel size | mode: fully CPU           | mode: partial gpu                       | mode: fully GPU       |
|:-----------|---------------------------|-----------------------------------------|:----------------------|
| 1mm        | system memory (RAM): 8 GB | RAM: 8 GB, graphics memory (VRAM): 2 GB | RAM: 8 GB, VRAM: 6 GB |
| 0.8mm      | RAM: 8 GB                 | RAM: 8 GB, VRAM: 2 GB                   | RAM: 8 GB, VRAM: 8 GB |
| 0.7mm      | RAM: 16 GB                | RAM: 16 GB, VRAM: 3 GB                  | RAM: 8 GB, VRAM: 8 GB |

The default device is the GPU. The view-aggregation device can be switched to CPU and requires less GPU memory. CPU-only processing ```--device cpu``` is much slower and not recommended.

Expert usage
------------
Individual modules and the surface pipeline can be run independently of the full pipeline script documented in this documentation. 
This is documented in READMEs in subfolders, for example: [whole brain segmentation only with FastSurferVINN](FastSurferCNN/README.md), [cerebellum sub-segmentation](CerebNet/README.md), [hypothalamic sub-segmentation](HypVINN/README.md), [corpus callosum analysis](CorpusCallosum/README.md) and [surface pipeline only (recon-surf)](recon_surf/README.md).

Specifically, the segmentation modules feature options for optimized parallelization of batch processing.

FreeSurfer Downstream Modules
-----------------------------
FreeSurfer provides several Add-on modules for downstream processing, such as subfield segmentation ( [hippocampus/amygdala](https://surfer.nmr.mgh.harvard.edu/fswiki/HippocampalSubfieldsAndNucleiOfAmygdala), [brainstem](https://surfer.nmr.mgh.harvard.edu/fswiki/BrainstemSubstructures), [thalamus](https://freesurfer.net/fswiki/ThalamicNuclei) and [hypothalamus](https://surfer.nmr.mgh.harvard.edu/fswiki/HypothalamicSubunits) ) as well as [TRACULA](https://surfer.nmr.mgh.harvard.edu/fswiki/Tracula). We now provide symlinks to the required files, as FastSurfer creates them with a different name (e.g. using "mapped" or "DKT" to make clear that these file are from our segmentation using the DKT Atlas protocol, and mapped to the surface). Most subfield segmentations require `wmparc.mgz` and work very well with FastSurfer,  so feel free to run those pipelines after FastSurfer. TRACULA requires `aparc+aseg.mgz` which we now link, but have not tested if it works, given that [DKT-atlas](https://mindboggle.readthedocs.io/en/latest/labels.html) merged a few labels. You should source FreeSurfer 7.3.2 to run these modules. 


Want to know more?
------------------
The DeepMI lab hosts an annual **FastSurfer course** at the German Center for Neurodegenerative Diseaes in Bonn, Germany. This is a 2.5-day, hands-on, introductory course on state-of-the-art deep-learning methods for fast and reliable neuroimage analysis. Participants will gain an understanding of modern methods for the analysis of structural brain images, learn how to run both the FastSurfer and FreeSurfer packages, and will know how to set up an analysis and work with the resulting outputs in the context of their own research projects. The course consists of lectures, demonstrations, practical exercises, and provides ample opportunities for discussions and informal exchange. The course typically takes place in **September**. Check out our [website](https://deep-mi.org/events) for details and current information!

Intended Use
------------
This software can be used to compute statistics from an MR image for research purposes. Estimates can be used to aggregate population data, compare groups etc. The data should not be used for clinical decision support in individual cases and, therefore, does not benefit the individual patient. Be aware that for a single image, produced results may be unreliable (e.g. due to head motion, imaging artefacts, processing errors etc). We always recommend to perform visual quality checks on your data, as also your MR-sequence may differ from the ones that we tested. No contributor shall be liable to any damages, see also our software [LICENSE](LICENSE). 

<!-- start of references -->
References
----------
If you use this for research publications, please cite:

_Henschel L, Conjeti S, Estrada S, Diers K, Fischl B, Reuter M, FastSurfer - A fast and accurate deep learning based neuroimaging pipeline, NeuroImage 219 (2020), 117012. https://doi.org/10.1016/j.neuroimage.2020.117012_

_Henschel L*, Kuegler D*, Reuter M. (*co-first). FastSurferVINN: Building Resolution-Independence into Deep Learning Segmentation Methods - A Solution for HighRes Brain MRI. NeuroImage 251 (2022), 118933. http://dx.doi.org/10.1016/j.neuroimage.2022.118933_

_Faber J*, Kuegler D*, Bahrami E*, et al. (*co-first). CerebNet: A fast and reliable deep-learning pipeline for detailed cerebellum sub-segmentation. NeuroImage 264 (2022), 119703. https://doi.org/10.1016/j.neuroimage.2022.119703_

_Estrada S, Kuegler D, Bahrami E, Xu P, Mousa D, Breteler MMB, Aziz NA, Reuter M. FastSurfer-HypVINN: Automated sub-segmentation of the hypothalamus and adjacent structures on high-resolutional brain MRI. Imaging Neuroscience 2023; 1 1–32. https://doi.org/10.1162/imag_a_00034_

_Pollak C, Diers K, Estrada S, Kuegler D, Reuter M, FastSurfer-CC: A robust, accurate, and comprehensive framework for corpus callosum morphometry, pre-print on arXiv: https://doi.org/10.48550/arXiv.2511.16471_


Stay tuned for updates and follow us on [X/Twitter](https://twitter.com/deepmilab).

<!-- start of acknowledgements -->
Acknowledgements
----------------
This project is partially funded by:
- [Chan Zuckerberg Initiative](https://chanzuckerberg.com/eoss/proposals/fastsurfer-ai-based-neuroimage-analysis-package/)
- [German Federal Ministry of Education and Research](https://www.gesundheitsforschung-bmbf.de/de/deepni-innovative-deep-learning-methoden-fur-die-rechnergestutzte-neuro-bildgebung-10897.php)

The recon-surf pipeline is largely based on [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferMethodsCitation).
