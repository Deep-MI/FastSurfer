# Manual Edits

We have noticed that FastSurfer segmentations and surface results are very robust and rarely require any manual edits. 
However, for your convenience, we allow manual edits in various stages of the FastSurfer pipeline.
These editing options include approaches that are inherited from FreeSurfer as well as some FastSurfer-specific editing options.

The provided editing options may be changed or extended in the future, also depending on requests from the community.
Furthermore, we invite users to [contribute](CONTRIBUTING.md) such changes and/or datasets of paired MRI images and edited files to improve FastSurfer's neural networks.  

## What can be edited?

Edits primarily affect FastSurfer's surface pipeline but some edits exist also during the segmentation steps.
To understand how edits affect different results and how to perform them, it is important to understand the order of processing steps in FastSurfer. 

## Pipeline overview

The FastSurfer pipeline comprises segmentation modules and the surface module.

The outputs of the relevant modules are:
1. The *asegdkt module*:
   - as the primary output: the whole brain segmentation (default name: `<subject_dir>/mri/aparc.DKTatlas+aseg.deep.mgz`)
   - as secondary outputs from the whole brain segmentation: 
     - the aseg without CC segmentation (`<subject_dir>/mri/aseg.auto_noCCseg.mgz`) and 
     - the brainmask (`<subject_dir>/mri/mask.mgz`).
2. The *biasfield module* uses the white matter segmentation (from aseg) and computes:
   - a bias field-corrected version of the conformed input image (`<subject_dir>/mri/orig_nu.mgz`) and
   - optionally may also perform a Talairach registration (`<subject_dir>/mri/transforms/talairach.(xfm|lta)`).
3. Other segmentation modules (e.g. the hypothalamus segmentation) use the biasfield corrected image as input.  
4. The *surface module* (*recon-surf*) generates surfaces and stats files based on outputs from the *asegdkt module*, including:
   1. a Talairach registration (`<subject_dir>/mri/transforms/talairach.(xfm|lta)`), if not already performed before,
   2. the WM segmentation (`<subject_dir>/mri/wm.mgz`) and filled version (`<subject_dir>/mri/filled.mgz`) to initialize surfaces and
   3. a brainmask (`<subject_dir>/mri/brain.finalsurfs.mgz`) to guide the positioning of the pial surfaces.

## Possible Edits

FastSurfer supports the following edits: 
1. Bias field corrected inputs (for improved image quality, not really an edit)
2. asegdkt_segfile: `<subject_dir>/mri/aparc.DKTatlas+aseg.deep.mgz` via `<subject_dir>/mri/aparc.DKTatlas+aseg.deep.manedit.mgz`
3. Talairach registration: `<subject_dir>/mri/transforms/talairach.xfm` (overwrites automatic results from `<subject_dir>/mri/transforms/talairach.auto.xfm`)
4. White matter segmentation: `<subject_dir>/mri/wm.mgz` and `<subject_dir>/mri/filled.mgz`
5. Pial placement: `<subject_dir>/mri/brain.finalsurfs.mgz` via `<subject_dir>/mri/brain.finalsurfs.manedit.mgz`

Note, as FastSurfer's surface pipeline is derived from FreeSurfer, some edit options and corresponding naming schemes are inherited from FreeSurfer.

## General process
Edits in this list require an update of the subject directory and "re-processing" with the `--edits` flag of `run_fastsurfer.sh`.

For example (including setup for native processing, see [Examples](EXAMPLES.md) for other processing options):

```bash
# Setup FASTSURFER and FREESURFER
export FASTSURFER_HOME=/path/to/fastsurfer
export FREESURFER_HOME=/path/to/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Define data directory
export SUBJECTS_DIR=/home/user/my_fastsurfer_analysis

# Run FastSurfer
$FASTSURFER_HOME/run_fastsurfer.sh \
    --sd $SUBJECTS_DIR --sid case_with_edits \
    --t1 $SUBJECTS_DIR/case_with_edits/mri/orig/001.mgz \
    --fs_license $FREESURFER_HOME/.license \
    --edits # more flags as you needed, e.g. --parallel --3T --threads 4
```

Note, a rerun of the segmentation pipeline, as in the command above, should not be harmful, but is only required if the [asegdkt_segfile](#asegdkt_segfile) was edited.
There, we can usually skip the segmentation step with
```bash
# Setup FASTSURFER and FREESURFER ... (see above)

# Run FastSurfer
$FASTSURFER_HOME/run_fastsurfer.sh \
    --sd $SUBJECTS_DIR --sid case_with_edits \
    --t1 $SUBJECTS_DIR/case_with_edits/mri/orig/001.mgz \
    --fs_license $FREESURFER_HOME/.license \
    --edits --surf_only # more flags as you needed, e.g. --parallel --3T --threads 4
```

## Bias field correction

This edit is "outside" of the FastSurfer pipeline and not really an edit.

### When to use 
The *asegdkt module* failed or produced unreliable segmentation maps, especially if bias fields are very strong as for example for 7T images. 
This can be detected by inspecting `<subject_dir>/mri/aparc.DKTatlas+aseg.deep.mgz`, `<subject_dir>/mri/aseg.auto_noCCseg.mgz` and `<subject_dir>/mri/mask.mgz`.

### What to do 
Perform bias field correction prior to segmentation (FastSurfer) and provide the bias corrected image as input. Sometimes, even FastSurfer's own bias field correction can help, which can be found in `<subject_dir>/mri/orig_nu.mgz`.
Alternatively, external bias field correction tools may help (e.g. *bias_correct* from SPM).
Run FastSurfer as a new run on this new input file.

For example:
- Step 1: Run FastSurfer to obrtain a bias field corrected image (not needed if you already processed with FastSurfer a first time): 
  ```bash
  # Setup FASTSURFER and FREESURFER ... (see above)
  
  # Run FastSurfer
  $FASTSURFER_HOME/run_fastsurfer.sh \
      --sd $SUBJECTS_DIR --sid case_bias_only \
      --t1 /path/to/the/original/T1.mgz \
      --seg_only --no_hypothal --no_cereb --threads 16
   ```
- Step 2: Run FastSurfer again, but this time input the bias field corrected image i.e ```orig_nu.mgz``` instead of original input image. The file ```orig_nu.mgz``` can be found in the output directory under the *mri* subfolder. The output produced from the second iteration should be saved in a different output directory for comparative analysis with the output produced in first iteration.
  ```bash
  # Setup FASTSURFER and FREESURFER ... (see above)

  # Run FastSurfer
  $FASTSURFER_HOME/run_fastsurfer.sh \
      --sd $SUBJECTS_DIR --sid case_bias_corrected \
      --t1 $SUBJECTS_DIR/case_bias_only/mri/orig_nu.mgz \
      --fs_license $FREESURFER_HOME/.license # more flags as you needed, e.g. --parallel --3T --threads 4
   ```
- Step 3: Compare and check if bias field correction fixed the issues:
  ```bash
  freeview $SUBJECTS_DIR/case_bias_only/mri/orig_nu.mgz $SUBJECTS_DIR/case_bias_corrected/mri/aparc.DKTatlas+aseg.deep.mgz $SUBJECTS_DIR/case_bias_only/mri/orig_nu.mgz $SUBJECTS_DIR/case_bias_corrected/mri/aparc.DKTatlas+aseg.deep.mgz
  ```

## asegdkt_segfile

### When to use 
(Minor) Segmentation errors such as over- and under-segmentation, of the gray and white matter, but also subcortical structures.
In particular, over- and under-segmentation of the brainmask, gray matter over segmentation into the dura, white matter undersegmentation in the cortex (causing missed gyri or cortical thickness overestimation).
In specific, such errors are inspected in `<subject_dir>/mri/aparc.DKTatlas+aseg.deep.mgz`, `<subject_dir>/mri/aseg.auto_noCCseg.mgz` and `<subject_dir>/mri/mask.mgz`.

### What to do
1. Copy `<subject_dir>/mri/aparc.DKTatlas+aseg.deep.mgz` to `<subject_dir>/mri/aparc.DKTatlas+aseg.deep.manedit.mgz`.
2. Open and edit `<subject_dir>/mri/aparc.DKTatlas+aseg.deep.manedit.mgz` fixing any errors.
3. [Re-run FastSurfer](#general-process) to update `<subject_dir>/mri/aseg.auto_noCCseg.mgz` and `<subject_dir>/mri/mask.mgz`.

<details><summary>Similar FreeSurfer edits</summary>

These are some edits that are possible in FreeSurfer and would be fixed by our asegdkt_segfile edits above:
1. Skull stripping: Edit `<subject_dir>/mri/brainmask.mgz` (delta to `brainmask.auto.mgz`)
2. Brain extraction: Edit `<subject_dir>/mri/brain.mgz` (delta to `brain.auto.mgz`)
3. Subcortical segmentation: `<subject_dir>/mri/aseg.presurf.mgz` (delta to `aseg.presurf.auto.mgz`)
4. CRS seeds: PONS, CC, LH, RH => seed files (`$subjdir/scripts/seed-(pons|cc|lh|rh|ws).crs.man.dat`)
5. watershed seed: seed file (`$subjdir/scripts/seed-ws.crs.man.dat`)

</details>

## Talairach registration

### When to use 
The estimated total intracranial volume is inaccurate.

### What to do 
1. Copy and edit `<subject_dir>/mri/transforms/talairach.xfm` replacing the old file.
2. [Re-run FastSurfer](#general-process) to update the eTIV value in all stats files.

See also: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/Talairach

<details><summary>Similar FreeSurfer edits</summary>

Talairach registration

</details>

## White matter segmentation

### When to use
Over- and/or under-segmentation of the white matter: voxels that should be white matter are excluded, or those that should not are included.

### What to do
Often, these errors should be fixed in [#asegdkt_segfile] `<subject_dir>/mri/aparc.DKTatlas+aseg.deep.manedit.mgz`, but if that is not possible, `<subject_dir>/mri/wm.mgz` and `<subject_dir>/mri/filled.mgz` can be edited to fix the initial white matter surface.

The manual label 255 indicates a voxel should be included in the white matter and a voxel labeled 1 should not.

See also: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/WhiteMatterEdits_freeview

<details><summary>Similar FreeSurfer edits</summary>

1. WM control points `$ControlPointsFile`: `<subject_dir>/tmp/control.dat`
2. in-file edits of the white matter segmentation: `<subject_dir>/mri/wm.mgz` (delta to `wm.auto.mgz`)
3. in-file edits of the filled Fill white matter segmentation: `<subject_dir>/mri/filled.mgz` (delta to `filled.auto.mgz`)

</details>

## Pial surface placement

### When to use
Over- and/or under-segmentation of the cortical gray matter: voxels that should be gray matter are excluded, or those that should not are included.

### What to do
Often, these errors should be fixed in [#asegdkt_segfile] `<subject_dir>/mri/aparc.DKTatlas+aseg.deep.manedit.mgz`, but if that is not possible, `<subject_dir>/mri/brain.finalsurfs.manedit.mgz` (overwriting values in `<subject_dir>/mri/brain.finalsurfs.mgz`) can be edited to fix the pial surface.
The manual label 255 indicates a voxel should be included in the gray matter and a voxel labeled 1 should not.

See also: https://surfer.nmr.mgh.harvard.edu/fswiki/Edits#brain.finalsurfs.mgz

<details><summary>Similar FreeSurfer edits</summary>

pial placement correction: `<subject_dir>/mri/brain.finalsurfs.manedit.mgz` (delta to `brain.finalsurfs.mgz`)

</details>

## Side effects of edits

Technically, all edits are changes to the processing pipeline and may cause systematic effects in the analysis.
Computational edits (e.g. [bias field correction](#bias-field-correction)) should be integrated systematically.
Other effects are impossible to avoid (edit carefully), often very small and hard to account for. 
It is recommended to analyze whether files with edits distort results in a specific direction, i.e. how do effect sizes change if manually edited cases are excluded?

We hope that this will help with (some of) your editing needs. 
Thanks for using FastSurfer. 
