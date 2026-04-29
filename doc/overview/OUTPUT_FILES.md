Output files
============
Segmentation module
-------------------
The segmentation module outputs the files shown in the table below. The two primary output files are the `aparc.DKTatlas+aseg.deep.mgz` file, which contains the FastSurfer segmentation of cortical and subcortical structures based on the DKT atlas, and the `aseg+DKT.stats` file, which contains summary statistics for these structures. Note, that the surface model (downstream) corrects these segmentations along the cortex with the created surfaces. So if the surface model is used, it is recommended to use the updated segmentations and stats (see below).

| directory | filename                     | module  | description                                                        |
|:----------|------------------------------|---------|--------------------------------------------------------------------|
| mri       | aparc.DKTatlas+aseg.deep.mgz | asegdkt | cortical and subcortical segmentation                              |
| mri       | aseg.auto_noCCseg.mgz        | asegdkt | simplified subcortical segmentation without corpus callosum labels |
| mri       | mask.mgz                     | asegdkt | brainmask                                                          |
| mri       | orig.mgz                     | asegdkt | conformed image                                                    |
| mri       | orig_nu.mgz                  | asegdkt | biasfield-corrected image                                          |
| mri/orig  | 001.mgz                      | asegdkt | original image                                                     |
| scripts   | deep-seg.log                 | asegdkt | logfile                                                            |
| stats     | aseg+DKT.stats               | asegdkt | table of cortical and subcortical segmentation statistics          |

Corpus Callosum module
----------------------
The Corpus Callosum module outputs the files in the table shown below. It creates detailed segmentations and shape analysis of the corpus callosum. For advanced output refer to the [FastSurfer-CC documentation](../scripts/fastsurfer_cc.rst).

| directory       | filename                       | module | description                                                                                                  |
|:----------------|--------------------------------|--------|--------------------------------------------------------------------------------------------------------------|
| mri             | callosum.CC.upright.mgz        | cc     | corpus callosum segmentation in upright space                                                                |
| mri             | callosum.CC.orig.mgz           | cc     | corpus callosum segmentation in conformed image orientation                                                  |
| mri             | callosum.CC.soft.mgz           | cc     | corpus callosum soft labels (in upright space)                                                               |
| mri             | fornix.CC.soft.mgz             | cc     | fornix soft labels (in upright space)                                                                        |
| mri             | background.CC.soft.mgz         | cc     | background soft labels (in upright space)                                                                    |
| mri             | upright_volume.mgz             | cc     | conformed image mapped to upright space (only with fastsurfer_cc.py `--upright_volume`)                                       |
| mri/transforms  | cc_up.lta                      | cc     | transform from conformed to upright space                                                                     |
| mri/transforms  | orient_volume.lta              | cc     | transform to standardized space                                                                              |
| stats           | callosum.CC.midslice.json      | cc     | measurements from the mid-sagittal slice (landmarks, area, thickness, etc.)                               |
| stats           | callosum.CC.all_slices.json    | cc     | comprehensive per-slice analysis       |
| qc_snapshots    | callosum.png                   | cc     | debug visualization of CC contours, AC, PC and thickness (only with run_fastsurfer.sh `--qc_snap`)                                       |
| qc_snapshots    | callosum_thickness.png         | cc     | 3D thickness visualization (only with run_fastsurfer.sh `--qc_snap`)                                                   |
| qc_snapshots    | corpus_callosum.html           | cc     | interactive 3D mesh visualization (only with run_fastsurfer.sh `--qc_snap`)                                                    |
| surf            | callosum.surf                  | cc     | 3D Corpus Callosum mesh in FreeSurfer surface format (open with freeview)              |
| surf            | callosum.thickness.w           | cc     | FreeSurfer overlay file containing thickness values (open with callosum.surf in freeview)                   |
| surf            | callosum.vtk                   | cc     | VTK format mesh file for 3D visualization                                        |

CerebNet module
---------------
The cerebellum module outputs the files in the table shown below. Unless switched off by the `--no_cereb` argument, this module is automatically run whenever the segmentation module is run. It adds two files, an image with the sub-segmentation of the cerebellum and a text file with summary statistics.

| directory | filename                   | module   | description                                 |
|:----------|----------------------------|----------|---------------------------------------------|
| mri       | cerebellum.CerebNet.nii.gz | cerebnet | cerebellum sub-segmentation                 |
| stats     | cerebellum.CerebNet.stats  | cerebnet | table of cerebellum segmentation statistics |

HypVINN module
--------------
The hypothalamus module outputs the files in the table shown below. Unless switched off by the `--no_hypothal` argument, this module is automatically run whenever the segmentation module is run. It adds three files, an image with the sub-segmentation of the hypothalamus and a text file with summary statistics.

| directory | filename                         | module  | description                                   |
|:----------|----------------------------------|---------|-----------------------------------------------|
| mri       | hypothalamus.HypVINN.nii.gz      | hypvinn | hypothalamus sub-segmentation                 |
| mri       | hypothalamus_mask.HypVINN.nii.gz | hypvinn | hypothalamus sub-segmentation mask            |
| stats     | hypothalamus.HypVINN.stats       | hypvinn | table of hypothalamus segmentation statistics |

If a T2 image is also passed, the following images are created.

| directory | filename      | module  | description                    |
|:----------|---------------|---------|--------------------------------|
| mri       | T2_nu.mgz     | hypvinn | biasfield-corrected T2 image   |
| mri       | T2_nu_reg.mgz | hypvinn | co-registered T2 to orig image |

Surface module
--------------
The surface module is run unless switched off by the `--seg_only` argument. It outputs a large number of files, which generally correspond to the FreeSurfer nomenclature and definition. A selection of important output files is shown in the table below, for the other files, we refer to the [FreeSurfer documentation](https://surfer.nmr.mgh.harvard.edu/fswiki). In general, the "mri" directory contains images, including segmentations, the "surf" folder contains surface files (geometries and vertex-wise overlay data), the "label" folder contains cortical parcellation labels, and the "stats" folder contains tabular summary statistics. Many files are available for the left ("lh") and right ("rh") hemisphere of the brain. Symbolic links are created to map FastSurfer files to their FreeSurfer equivalents, which may need to be present for further processing (e.g., with FreeSurfer downstream modules).

After running this module, some of the initial segmentations and corresponding volume estimates are fine-tuned (e.g., surface-based partial volume correction, addition of corpus callosum labels). Specifically, this concerns the `aseg.mgz `, `aparc.DKTatlas+aseg.mapped.mgz`, `aparc.DKTatlas+aseg.deep.withCC.mgz`, which were originally created by the segmentation module or have earlier versions resulting from that module.

The primary output files are pial, white, and inflated surface files, the thickness overlay files, and the cortical parcellation (annotation) files. The preferred way of assessing this output is the [FreeView](https://surfer.nmr.mgh.harvard.edu/fswiki/FreeviewGuide) software. Summary statistics for volume and thickness estimates per anatomical structure are reported in the stats files, in particular the `aseg.stats`, and the left and right `aparc.DKTatlas.mapped.stats` files.

| directory | filename                                                       | module  | description                                                                                  |
|:----------|----------------------------------------------------------------|---------|----------------------------------------------------------------------------------------------|
| mri       | aparc.DKTatlas+aseg.deep.withCC.mgz                            | surface | cortical and subcortical segmentation incl. corpus callosum after running the surface module |
| mri       | aparc.DKTatlas+aseg.mapped.mgz                                 | surface | cortical and subcortical segmentation after running the surface module                       |
| mri       | aparc.DKTatlas+aseg.mgz                                        | surface | symlink to aparc.DKTatlas+aseg.mapped.mgz                                                    |
| mri       | aparc+aseg.mgz                                                 | surface | symlink to aparc.DKTatlas+aseg.mapped.mgz                                                    |
| mri       | aseg.mgz                                                       | surface | subcortical segmentation after running the surface module                                    |
| mri       | wmparc.DKTatlas.mapped.mgz                                     | surface | white matter parcellation                                                                    |
| mri       | wmparc.mgz                                                     | surface | symlink to wmparc.DKTatlas.mapped.mgz                                                        |
| surf      | lh.area, rh.area                                               | surface | surface area overlay file                                                                    |
| surf      | lh.curv, rh.curv                                               | surface | curvature overlay file                                                                       |
| surf      | lh.inflated, rh.inflated                                       | surface | inflated cortical surface                                                                    |
| surf      | lh.pial, rh.pial                                               | surface | pial surface                                                                                 |
| surf      | lh.thickness, rh.thickness                                     | surface | cortical thickness overlay file                                                              |
| surf      | lh.volume, rh.volume                                           | surface | gray matter volume overlay file                                                              |
| surf      | lh.white, rh.white                                             | surface | white matter surface                                                                         |
| label     | lh.aparc.DKTatlas.annot, rh.aparc.DKTatlas.annot               | surface | symlink to lh.aparc.DKTatlas.mapped.annot                                                    |
| label     | lh.aparc.DKTatlas.mapped.annot, rh.aparc.DKTatlas.mapped.annot | surface | annotation file for cortical parcellations, mapped from ASEGDKT segmentation to the surface  |
| stats     | aseg.stats                                                     | surface | table of cortical and subcortical segmentation statistics after running the surface module   |
| stats     | lh.aparc.DKTatlas.mapped.stats, rh.aparc.DKTatlas.mapped.stats | surface | table of cortical parcellation statistics, mapped from ASEGDKT segmentation to the surface   |
| stats     | lh.curv.stats, rh.curv.stats                                   | surface | table of curvature statistics                                                                |
| stats     | wmparc.DKTatlas.mapped.stats                                   | surface | table of white matter segmentation statistics                                                |
| scripts   | recon-all.log                                                  | surface | logfile                                                                                      |

Lesion Inpainting Tool (LIT, optional)
--------------------------------------
When `--lesion_mask <path to file>` is provided, FastSurfer wraps the segmentation and surface
pipelines with lesion inpainting using LIT. The extension is currently experimental. It inpaints
the lesion region, runs the requested FastSurfer modules on the inpainted image, and then maps the
lesion back into the resulting outputs. The current LIT postprocessing workflow updates the primary
FastSurfer files in place and keeps the original pre-lesion outputs either as `.lit` backups or,
for some surface-derived files, in the original `.mapped.*` files.

For lesion mask requirements, see the [FastSurfer-LIT module documentation](modules/LIT.md#lesion-mask-requirements).

### Inpainting Outputs
These are the key files created during the initial inpainting stage. FastSurfer with LIT writes
these outputs directly into the standard subject directory layout.

| directory | filename | module | description |
|:----------|----------|--------|-------------|
| mri | inpainted.lit.nii.gz | lit | inpainted T1 image used for downstream processing |
| mri | mask.lit.nii.gz | lit | processed lesion mask in FastSurfer image space, after optional preprocessing |
| mri/orig | mask.lit.nii.gz | lit | original lesion mask stored in the subject directory |
| mri/orig | inpainting_original_image.lit.nii.gz | lit | conformed original image used internally by LIT |
| mri/orig | inpainting_masked_image.lit.nii.gz | lit | conformed masked image used internally by LIT |
| scripts | inpainting_*.lit.png | lit | preview images from the inpainting step |

### Postprocessing MRI Outputs
These files contain the lesion-integrated segmentations. LIT overwrites the primary FastSurfer outputs and stores the pre-lesion versions as `.lit` backups.

| directory | filename | module | description |
|:----------|----------|--------|-------------|
| mri | aparc.DKTatlas+aseg.deep.mgz | lit | lesion-integrated whole-brain segmentation |
| mri | aparc.DKTatlas+aseg.deep.lit.mgz | lit | backup of the pre-lesion whole-brain segmentation |
| mri | aseg.auto_noCCseg.mgz | lit | lesion-integrated subcortical segmentation used for VINN statistics |
| mri | aseg.auto_noCCseg.lit.mgz | lit | backup of the pre-lesion subcortical segmentation |
| mri | cerebellum.CerebNet.nii.gz | lit | lesion-integrated cerebellum segmentation when CerebNet is available |
| mri | cerebellum.CerebNet.lit.nii.gz | lit | backup of the pre-lesion cerebellum segmentation |
| mri | hypothalamus.HypVINN.nii.gz | lit | lesion-integrated hypothalamus segmentation when HypVINN is available |
| mri | hypothalamus.HypVINN.lit.nii.gz | lit | backup of the pre-lesion hypothalamus segmentation |

### Postprocessing Statistics and Reports
LIT regenerates the relevant stats files after lesion mapping, keeps the pre-lesion versions as `.lit` backups where applicable, and writes lesion-specific reports.

| directory | filename | module | description |
|:----------|----------|--------|-------------|
| stats | lesion_impact_summary.yaml | lit | machine-readable summary of affected brain regions |
| stats | aparc.DKTatlas+aseg.lesion_report.txt | lit | report of volumetric structures affected by the lesion |
| stats | aseg.lesion_report.txt | lit | report of affected structures in the FreeSurfer aseg segmentation |
| stats | aseg+DKT.VINN.stats | lit | lesion-integrated whole-brain/VINN summary statistics |
| stats | aseg+DKT.VINN.lit.stats | lit | backup of the pre-lesion whole-brain/VINN statistics |
| stats | aseg.VINN.stats | lit | lesion-integrated subcortical VINN statistics |
| stats | aseg.VINN.lit.stats | lit | backup of the pre-lesion subcortical VINN statistics |
| stats | cerebellum.CerebNet.stats | lit | lesion-integrated cerebellum statistics when CerebNet is available |
| stats | cerebellum.CerebNet.lit.stats | lit | backup of the pre-lesion cerebellum statistics |
| stats | hypothalamus.HypVINN.stats | lit | lesion-integrated hypothalamus statistics when HypVINN is available |
| stats | hypothalamus.HypVINN.lit.stats | lit | backup of the pre-lesion hypothalamus statistics |

### Surface-based Outputs
If the surface pipeline is run, LIT also updates the relevant surface annotations and stats. The
public annotation paths are kept at the standard FreeSurfer names, while the preserved pre-lesion
surface stats remain in the corresponding `.mapped.stats` files.

| directory | filename | module | description |
|:----------|----------|--------|-------------|
| label | {lh,rh}.aparc.DKTatlas.annot | lit | cortical parcellation with lesion projected onto the surface; symlink to `{lh,rh}.aparc.DKTatlas.mapped.annot` |
| label | {lh,rh}.aparc.DKTatlas.lit.annot | lit | pre-lesion cortical parcellation; symlink to `{lh,rh}.aparc.DKTatlas.mapped.lit.annot` |
| stats | {lh,rh}.aparc.DKTatlas.stats | lit | lesion-integrated cortical surface statistics |
| stats | {lh,rh}.aparc.DKTatlas.mapped.stats | lit | backup of the pre-lesion cortical surface statistics |
| stats | {lh,rh}.aparc.DKTatlas.anatomy_report.txt | lit | report of cortical structures affected by the lesion |

Longitudinal Processing
-----------------------
When running the [longitudinal pipeline](LONG.md) the output will be as above for the individual time point directories. Note that the templateID directory for the within-subject template will not contain all files and usually is not looked at or analyzed, as it represents an intermediate step in the longitudinal pipeline. 
