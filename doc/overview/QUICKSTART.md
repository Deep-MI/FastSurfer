Quick Start
===========
This example segments one brain MRI in a few minutes. Choose how you run FastSurfer: in a container on Linux
(Apptainer, Singularity or Docker), with the [macOS package](install/MACOS.md), or without installing anything in
[Google Colab](#google-colab). Other systems are covered in the [installation guide](INSTALL.md).

Segment an example image
------------------------
With a GPU (8 GB of graphics memory) this takes well under a minute; without one, FastSurfer uses the CPU and takes
a few minutes longer. You can use your own T1-weighted full head MRI (0.7 to 1 mm voxel sizes, no preprocessing,
`.nii`, `.nii.gz` or `.mgz`), or the example image the commands download.

`````{tab-set}

````{tab-item} Apptainer / Singularity
```bash
# 0. Create a directory for this test
mkdir fastsurfer_test
cd fastsurfer_test

# 1. Download the docker image and create the singularity image
#    (do this only the first time)
#    It will produce the fastsurfer-{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}.sif singularity image
#    file in $HOME/my_singularity_images
mkdir -p $HOME/my_singularity_images
singularity build \
    $HOME/my_singularity_images/fastsurfer-{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}.sif \
    docker://deepmi/fastsurfer:{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}

# 2. Download an example brain MRI (if you don't have your own)
#    If you have your own, copy it to this directory and adjust
#    the filename after --t1 below.
curl -k \
    https://surfer.nmr.mgh.harvard.edu/pub/data/tutorial_data/buckner_data/tutorial_subjs/140/mri/orig.mgz \
    -o "./140_orig.mgz"

# 3. Run FastSurfer (full brain segmentation only)
singularity exec --nv \
                 --no-mount home,cwd -e \
                 -B "$PWD" \
                 $HOME/my_singularity_images/fastsurfer-{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}.sif \
                 /fastsurfer/run_fastsurfer.sh \
                 --t1 "$PWD/140_orig.mgz" \
                 --sid subjectX --sd "$PWD" \
                 --seg_only --no_biasfield --no_cereb --no_hypothal
```
````

````{tab-item} Docker
```bash
# 0. Create a directory for this test
mkdir fastsurfer_test
cd fastsurfer_test

# 1. Download an example brain MRI (if you don't have your own)
#    If you have your own, copy it to this directory and adjust
#    the filename after --t1 below.
curl -k \
    https://surfer.nmr.mgh.harvard.edu/pub/data/tutorial_data/buckner_data/tutorial_subjs/140/mri/orig.mgz \
    -o "./140_orig.mgz"

# 2. Run FastSurfer (full brain segmentation only)
docker run --gpus all -v "$PWD:$PWD" \
                      --rm --user $(id -u):$(id -g) \
                      deepmi/fastsurfer:{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }} \
                      --t1 "$PWD/140_orig.mgz" \
                      --sid subjectX --sd "$PWD" \
                      --seg_only --no_biasfield --no_cereb --no_hypothal
```
````

````{tab-item} macOS package
[Install the package](install/MACOS.md), start the FastSurfer app from your Applications folder, and run these
commands in the console it opens:

```bash
# 0. Create a directory for this test
mkdir fastsurfer_test
cd fastsurfer_test

# 1. Download an example brain MRI (if you don't have your own)
#    If you have your own, copy it to this directory and adjust
#    the filename after --t1 below.
curl -k \
    https://surfer.nmr.mgh.harvard.edu/pub/data/tutorial_data/buckner_data/tutorial_subjs/140/mri/orig.mgz \
    -o "./140_orig.mgz"

# 2. Run FastSurfer (full brain segmentation only)
run_fastsurfer.sh --t1 "$PWD/140_orig.mgz" \
    --sid subjectX --sd "$PWD" \
    --seg_only --no_biasfield --no_cereb --no_hypothal
```
````

`````

That's it, it will run the full brain segmentation. For speed, we switched off the cerebellum and hypothalamic sub-segmentation (would add a couple minutes).
We also switched off the bias field correction, which is used to compute partial volume estimates for the statsfiles, so you might want to switch it on again if you want the volume statistics text file (under ```subjectX/stats```).
Also if you need the estimated total intracranial volume for correcting the stats, you would either need to run the surface stream or switch on the Talairach registration with
`--tal_reg` in the segmentation module. For the full surface stream, just remove the ```--seg_only``` and you need a [FreeSurfer license file](INSTALL.md#freesurfer-license) and pass it into the container, as described in more detail later.

You will find the full brain segmentation in ```./subjectX/mri/aparc.DKTatlas+aseg.deep.mgz``` in FreeSurfer's MGZ file format. To convert it back to nifti (if you prefer), run `nib-convert`, which comes with FastSurfer, for example with the Singularity image (in the macOS console, call `nib-convert` directly):

```bash
# Convert mgz to nifti
singularity exec --nv \
                 --no-mount home -e \
                 -B "$PWD" \
                 $HOME/my_singularity_images/fastsurfer-{{ CUDA_STRING }}-v{{ FASTSURFER_VERSION }}.sif \
                 nib-convert "$PWD/subjectX/mri/aparc.DKTatlas+aseg.deep.mgz" \
                             "$PWD/subjectX/mri/aparc.DKTatlas+aseg.deep.nii.gz"
```

and find the segmentation in ```./subjectX/mri/aparc.DKTatlas+aseg.deep.nii.gz```. If you have FreeSurfer installed, just use FreeView to look at the result (or really any other image viewer):

```bash
# FreeView
freeview -v 140_orig.mgz \
    subjectX/mri/aparc.DKTatlas+aseg.deep.mgz:colormap=lut:opacity=0.2
```

Other interesting outputs of the segmentation are the ```aseg.auto_noCCseg.mgz``` containing a reduced segmentation according to FreeSurfer's aseg (no cortical sub-division and no corpus callosum, which is added later). Also ```mask.mgz``` can come in handy if you need a brainmask. And you get all of this within a few seconds (including startup of singularity or docker it is **20 sec** in total with a GeForce RTX 4080, **40 sec** with a Quadro RTX 4000 or Titan XP, CPU-only processing takes about **5 minutes** longer).

Google Colab
------------
You can also run FastSurfer in the cloud with Google Colab.

In order to use the notebooks, simply click on the link or optimally the google colab icon displayed at the top of the page. This way, the plots will be rendered correctly. If you have a Google account, you can interactively execute the run cells. Without a google account you can see the files and outputs generated by the last run.

### Notebook 1 - Quick and Easy - FastSurfer Segmentation with three clicks
__[Notebook 1](https://colab.research.google.com/github/Deep-MI/FastSurfer/blob/stable/Tutorial/Tutorial_FastSurferCNN_QuickSeg.ipynb)__ contains a super quick and easy scenario in which you can run FastSurferCNN in just three clicks. You do not need any programming experience to get a segmentation in less than 60 s!

### Notebook 2 - Complete FastSurfer Tutorial
__[Notebook 2](https://colab.research.google.com/github/Deep-MI/FastSurfer/blob/stable/Tutorial/Complete_FastSurfer_Tutorial.ipynb)__ is an extended version of the first one with information about how to set up FastSurfer on your local machine. It points to the installation guide and includes examples of how to run FastSurfer, and how to visualize and quality control your data.

After a quick introduction, it covers three use cases:
- Use case 1: Quick and Easy - FastSurfer Segmentation with three clicks (same as the first notebook)
- Use case 2: Quick and a bit more advanced - Segmentation with FastSurfer on your local machine
- Use case 3: Surface models, Thickness maps and more: FastSurfer's recon-surf command

In addition, there is a small section covering [fsqc](https://github.com/Deep-MI/fsqc) called "Bonus - Quality analysis using fsqc".
